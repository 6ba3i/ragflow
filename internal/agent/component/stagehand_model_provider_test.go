package component

import (
	"context"
	"encoding/json/v2"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ragflow/internal/common"
	"ragflow/internal/entity/models"
)

func TestStagehandModelServerUsesProviderDriverProtocols(t *testing.T) {
	prev := common.AllowAnyHostForTest
	common.AllowAnyHostForTest = true
	t.Cleanup(func() { common.AllowAnyHostForTest = prev })

	t.Run("openai-compatible", func(t *testing.T) {
		var sawModel, sawAuth bool
		provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/chat/completions" {
				t.Errorf("path=%q, want /chat/completions", r.URL.Path)
			}
			sawAuth = r.Header.Get("Authorization") == "Bearer provider-key"
			var body struct {
				Model string           `json:"model"`
				Tools []map[string]any `json:"tools"`
			}
			if err := json.UnmarshalRead(r.Body, &body); err != nil {
				t.Errorf("decode provider request: %v", err)
			}
			sawModel = body.Model == "glm-4-flash" && len(body.Tools) == 1
			_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"goto","arguments":"{\"url\":\"https://example.com\"}"}}]},"finish_reason":"tool_calls"}]}`)
		}))
		defer provider.Close()

		key := "provider-key"
		modelName := "glm-4-flash"
		model := models.NewChatModel(
			models.NewOpenAIModel(map[string]string{"default": provider.URL}, models.URLSuffix{Chat: "chat/completions"}),
			&modelName,
			&models.APIConfig{ApiKey: &key},
		)
		result := postStagehandModel(t, model)
		if !sawAuth || !sawModel {
			t.Fatalf("OpenAI driver request lost provider config: auth=%v model=%v", sawAuth, sawModel)
		}
		if result["finish_reason"] != "tool_calls" {
			t.Fatalf("finish_reason=%v, want tool_calls", result["finish_reason"])
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		var sawModel, sawAuth, sawTool bool
		provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/messages" {
				t.Errorf("path=%q, want /v1/messages", r.URL.Path)
			}
			sawAuth = r.Header.Get("x-api-key") == "provider-key" &&
				r.Header.Get("anthropic-version") != ""
			var body struct {
				Model string           `json:"model"`
				Tools []map[string]any `json:"tools"`
			}
			if err := json.UnmarshalRead(r.Body, &body); err != nil {
				t.Errorf("decode provider request: %v", err)
			}
			sawModel = body.Model == "claude-sonnet-4-5"
			if len(body.Tools) == 1 {
				sawTool = body.Tools[0]["name"] == "goto" && body.Tools[0]["input_schema"] != nil
			}
			_, _ = fmt.Fprint(w, `{"id":"msg-1","content":[{"type":"tool_use","id":"call-1","name":"goto","input":{"url":"https://example.com"}}],"usage":{"input_tokens":2,"output_tokens":3}}`)
		}))
		defer provider.Close()

		key := "provider-key"
		modelName := "claude-sonnet-4-5"
		model := models.NewChatModel(
			models.NewAnthropicModel(map[string]string{"default": provider.URL}, models.URLSuffix{Chat: "v1/messages"}),
			&modelName,
			&models.APIConfig{ApiKey: &key},
		)
		result := postStagehandModel(t, model)
		if !sawAuth || !sawModel || !sawTool {
			t.Fatalf("Anthropic driver request lost provider config: auth=%v model=%v tool=%v", sawAuth, sawModel, sawTool)
		}
		if result["finish_reason"] != "tool_calls" {
			t.Fatalf("finish_reason=%v, want tool_calls", result["finish_reason"])
		}
	})
}

func postStagehandModel(t *testing.T, model *models.ChatModel) map[string]any {
	t.Helper()
	baseURL, token, closeServer, err := startStagehandModelServer(t.Context(), model)
	if err != nil {
		t.Fatal(err)
	}
	defer closeServer()

	body := `{"messages":[{"role":"user","content":"browse"}],"tools":[{"type":"function","function":{"name":"goto","description":"Navigate","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"goto"}}}`
	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, baseURL+"/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+token)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("bridge HTTP %d", response.StatusCode)
	}
	var result struct {
		Choices []struct {
			Message      map[string]any `json:"message"`
			FinishReason string         `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.UnmarshalRead(response.Body, &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("choices=%#v", result.Choices)
	}
	return map[string]any{
		"finish_reason": result.Choices[0].FinishReason,
		"message":       result.Choices[0].Message,
	}
}
