package component

import (
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	stagehand "github.com/browserbase/stagehand-go/v3"
	"github.com/browserbase/stagehand-go/v3/option"

	"ragflow/internal/common"
	"ragflow/internal/entity/models"
)

func TestStagehandRuntimeSharedModel(t *testing.T) {
	var bridgeURLs, tokens []string
	model := models.NewChatModel(stagehandChatDriver{chat: func(context.Context, []models.Message, *models.ChatConfig) (*models.ChatResponse, error) {
		return &models.ChatResponse{Answer: new("actual page title")}, nil
	}}, new("provider-native-model"), nil)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(request.URL.Path, "/start"):
			_, _ = fmt.Fprint(w, `{"success":true,"data":{"sessionId":"s"}}`)
		case strings.HasSuffix(request.URL.Path, "/agentExecute"):
			var input struct {
				AgentConfig struct {
					Mode  string `json:"mode"`
					Model struct {
						BaseURL   string `json:"baseURL"`
						APIKey    string `json:"apiKey"`
						ModelName string `json:"modelName"`
					} `json:"model"`
				} `json:"agentConfig"`
			}
			if err := json.UnmarshalRead(request.Body, &input); err != nil {
				t.Error(err)
				return
			}
			config := input.AgentConfig.Model
			if input.AgentConfig.Mode != "dom" || config.ModelName != "openai/ragflow" {
				t.Errorf("unexpected wire config: %+v", input)
			}
			bridgeURLs, tokens = append(bridgeURLs, config.BaseURL), append(tokens, config.APIKey)
			call, _ := http.NewRequestWithContext(request.Context(), http.MethodPost, config.BaseURL+"/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"read page"}]}`))
			call.Header.Set("Authorization", "Bearer "+config.APIKey)
			result, err := http.DefaultClient.Do(call)
			if err != nil {
				t.Error(err)
				return
			}
			defer result.Body.Close()
			if result.StatusCode != http.StatusOK {
				t.Errorf("bridge HTTP %d", result.StatusCode)
			}
			_, _ = fmt.Fprint(w, `{"success":true,"data":{"result":{"success":true,"completed":true,"message":"actual page title"}}}`)
		case strings.HasSuffix(request.URL.Path, "/end"):
			_, _ = fmt.Fprint(w, `{"success":true}`)
		default:
			http.NotFound(w, request)
		}
	}))
	defer server.Close()
	runtime := newStagehandRuntime(time.Hour, 0, time.Minute)
	defer runtime.Close()
	for _, tenant := range []string{"a", "b"} {
		entry := &stagehandClientEntry{client: stagehand.NewClient(option.WithBaseURL(server.URL), option.WithMaxRetries(0))}
		entry.lastUsedAt.Store(time.Now().UnixNano())
		runtime.cache.Store(fmt.Sprintf("resolved:%q:%q", tenant, "model-id"), entry)
	}
	for _, tenant := range []string{"a", "a", "b"} {
		result, err := runtime.RunTask(t.Context(), RunTaskRequest{TenantID: tenant, LLMID: "model-id", ChatModel: model, Instruction: "read page"})
		if err != nil || result != "actual page title" {
			t.Fatalf("RunTask = %q, %v", result, err)
		}
	}
	if cacheSize(runtime) != 2 || len(tokens) != 3 || tokens[0] == tokens[1] {
		t.Fatalf("bridge credentials leaked into subprocess cache: %d entries, %d calls", cacheSize(runtime), len(tokens))
	}
	for _, baseURL := range bridgeURLs {
		response, err := http.Get(baseURL)
		if err == nil {
			response.Body.Close()
			t.Fatal("bridge listener survived task completion")
		}
	}
}

type stagehandChatDriver struct {
	models.ModelDriver
	chat func(context.Context, []models.Message, *models.ChatConfig) (*models.ChatResponse, error)
}

func (d stagehandChatDriver) ChatWithMessages(ctx context.Context, _ string, messages []models.Message, _ *models.APIConfig, config *models.ChatConfig, _ *common.ModelUsage) (*models.ChatResponse, error) {
	return d.chat(ctx, messages, config)
}

func TestStagehandModelServer(t *testing.T) {
	for _, operation := range []string{"tools", "schema", "provider_error", "missing_schema_result"} {
		t.Run(operation, func(t *testing.T) {
			called := false
			driver := stagehandChatDriver{chat: func(ctx context.Context, messages []models.Message, config *models.ChatConfig) (*models.ChatResponse, error) {
				called = true
				if len(messages) != 2 || messages[0].Role != "system" || messages[1].ToolCallID != "previous-call" {
					t.Fatalf("messages lost their roles/tool metadata: %#v", messages)
				}
				if config.Temperature == nil || *config.Temperature != 0.2 || config.MaxTokens == nil || *config.MaxTokens != 512 {
					t.Fatalf("generation options lost: %#v", config)
				}
				if operation == "provider_error" {
					return nil, errors.New("provider unavailable")
				}
				if operation == "schema" {
					choice, ok := config.ToolChoiceValue.(map[string]any)
					if !ok || choice["type"] != "function" {
						t.Fatalf("schema was not enforced through tools: %#v", config)
					}
					return &models.ChatResponse{ToolCalls: []map[string]any{{"id": "structured", "type": "function", "function": map[string]any{"name": "ragflow_structured_output", "arguments": `{"title":"actual title"}`}}}}, nil
				}
				if operation == "missing_schema_result" {
					return &models.ChatResponse{Answer: new("not JSON")}, nil
				}
				if config.Tools == nil || config.ToolChoiceValue == nil {
					t.Fatal("tool configuration lost")
				}
				return &models.ChatResponse{Answer: new("answer"), ReasonContent: new("private reasoning"), ToolCalls: []map[string]any{{"id": "next", "type": "function", "function": map[string]any{"name": "goto", "arguments": `{"url":"https://example.com"}`}}}, Usage: &models.TokenUsage{PromptTokens: 12, CompletionTokens: 4, TotalTokens: 16}}, nil
			}}
			model := models.NewChatModel(driver, new("actual-provider-model"), &models.APIConfig{})
			baseURL, token, closeServer, err := startStagehandModelServer(t.Context(), model)
			if err != nil {
				t.Fatal(err)
			}
			defer closeServer()
			body := `{"messages":[{"role":"system","content":"instructions"},{"role":"tool","content":"page","tool_call_id":"previous-call"}],"temperature":0.2,"max_tokens":512,"tools":[{"type":"function","function":{"name":"goto","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"goto"}}}`
			if operation == "schema" || operation == "missing_schema_result" {
				body = `{"messages":[{"role":"system","content":"instructions"},{"role":"tool","content":"page","tool_call_id":"previous-call"}],"temperature":0.2,"max_tokens":512,"response_format":{"type":"json_schema","json_schema":{"name":"result","schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}}}}`
			}
			request, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, baseURL+"/chat/completions", strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer "+token)
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			if !called {
				t.Fatal("shared model driver was not invoked")
			}
			if operation == "provider_error" || operation == "missing_schema_result" {
				if response.StatusCode != http.StatusBadGateway {
					t.Fatalf("failure hidden: HTTP %d", response.StatusCode)
				}
				return
			}
			if response.StatusCode != http.StatusOK {
				t.Fatalf("HTTP %d", response.StatusCode)
			}
			var result struct {
				Choices []struct {
					Message      models.Message `json:"message"`
					FinishReason string         `json:"finish_reason"`
				} `json:"choices"`
				Usage models.TokenUsage `json:"usage"`
			}
			if err := json.UnmarshalRead(response.Body, &result); err != nil {
				t.Fatal(err)
			}
			if len(result.Choices) != 1 {
				t.Fatalf("invalid completion: %#v", result)
			}
			if operation == "schema" {
				if result.Choices[0].Message.Content != `{"title":"actual title"}` || len(result.Choices[0].Message.ToolCalls) != 0 {
					t.Fatalf("schema response not translated: %#v", result)
				}
			} else if result.Choices[0].FinishReason != "tool_calls" || len(result.Choices[0].Message.ToolCalls) != 1 || result.Usage.TotalTokens != 16 {
				t.Fatalf("completion metadata lost: %#v", result)
			}
		})
	}
}

func TestStagehandModelServerRequiresPrivateToken(t *testing.T) {
	model := models.NewChatModel(stagehandChatDriver{chat: func(context.Context, []models.Message, *models.ChatConfig) (*models.ChatResponse, error) {
		t.Fatal("unauthenticated model invocation")
		return nil, nil
	}}, new("model"), nil)
	baseURL, _, closeServer, err := startStagehandModelServer(t.Context(), model)
	if err != nil {
		t.Fatal(err)
	}
	defer closeServer()
	request, _ := http.NewRequestWithContext(t.Context(), http.MethodPost, baseURL+"/chat/completions", strings.NewReader(`{}`))
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusUnauthorized {
		t.Fatalf("HTTP %d", response.StatusCode)
	}
}
