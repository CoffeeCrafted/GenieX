// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

type recorded struct {
	events []string
	fields []map[string]any
}

func (r *recorded) emit(eventType string, fields map[string]any) {
	r.events = append(r.events, eventType)
	r.fields = append(r.fields, fields)
}

func TestOutputBuilderEventOrder(t *testing.T) {
	var rec recorded
	b := newOutputBuilder(rec.emit, nil, nil)
	b.thinking("Let me ")
	b.thinking("count.")
	b.content("\n\n") // the newline after thinking opens nothing on its own
	b.content("1, 2, ")
	b.content("3")
	b.finish("completed")

	equal(t, rec.events, []string{
		"response.output_item.added", // reasoning
		"response.content_part.added",
		"response.reasoning.delta",
		"response.reasoning.delta",
		"response.reasoning.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.output_item.added", // message
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
	})
	if len(b.items) != 2 {
		t.Fatalf("items = %d, want a reasoning and a message item", len(b.items))
	}
	msg := b.items[1].(messageItem)
	if msg.Content[0].Text != "1, 2, 3" || msg.Status != "completed" {
		t.Errorf("message = %+v, want text without the leading newlines, completed", msg)
	}
	if done := rec.fields[11]; done["text"] != "1, 2, 3" || done["output_index"] != 1 || done["content_index"] != 0 {
		t.Errorf("output_text.done fields = %v", done)
	}
	if b.reasoning != 2 {
		t.Errorf("reasoning tokens = %d, want 2", b.reasoning)
	}
}

func TestOutputBuilderFunctionCalls(t *testing.T) {
	var rec recorded
	b := newOutputBuilder(rec.emit, map[string]bool{"get_weather": true}, nil)
	b.content("\n")
	b.toolCall("get_weather", `{"city":"Paris"}`)
	b.toolCall("delete_files", `{}`) // not allowed by tool_choice
	b.finish("completed")

	equal(t, rec.events, []string{
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
	})
	call := b.items[0].(functionCallItem)
	if call.Name != "get_weather" || call.Arguments != `{"city":"Paris"}` || call.Status != "completed" || call.CallID == "" {
		t.Errorf("call = %+v", call)
	}
	added := rec.fields[0]["item"].(functionCallItem)
	if added.Status != "in_progress" || added.Name != "get_weather" || added.Arguments != "" {
		t.Errorf("output_item.added item = %+v, want in_progress with the name set", added)
	}
	if len(b.stored) != 1 || b.stored[0].CallID != call.CallID {
		t.Errorf("stored = %+v, want the call kept for previous_response_id", b.stored)
	}
}

func TestOutputBuilderMaxToolCalls(t *testing.T) {
	one := int64(1)
	b := newOutputBuilder(nil, nil, &one)
	b.toolCall("a", "{}")
	b.toolCall("b", "{}")
	b.finish("completed")
	if len(b.items) != 1 {
		t.Errorf("items = %d, want max_tool_calls to keep one", len(b.items))
	}
}

func TestOutputBuilderIncompleteMessage(t *testing.T) {
	b := newOutputBuilder(nil, nil, nil)
	b.content("The answer is")
	b.finish("incomplete")
	if msg := b.items[0].(messageItem); msg.Status != "incomplete" {
		t.Errorf("status = %q, want incomplete", msg.Status)
	}
}

func TestChatErrorBodyIsUnchanged(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	writeAPIError(c, http.StatusNotFound, map[string]any{"error": "model not found"})
	if got := w.Body.String(); got != `{"error":"model not found"}` {
		t.Errorf("chat error body = %s", got)
	}
}

func TestResponsesErrorBodyFollowsTheSpec(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	useResponsesErrors(c, httpResponsesErrors(c))
	writeAPIError(c, http.StatusInternalServerError, map[string]any{"error": "load failed", "code": -100201})
	var body struct {
		Error map[string]any `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"type": "server_error", "message": "load failed", "code": "-100201", "param": nil}
	for k, v := range want {
		if body.Error[k] != v {
			t.Errorf("error.%s = %v, want %v (body %s)", k, body.Error[k], v, w.Body.String())
		}
	}
}

// TestGenerationFailedIsOneRule pins the error every transport and compaction
// answer with for the same generation error.
func TestGenerationFailedIsOneRule(t *testing.T) {
	for _, err := range []error{nil, geniex_sdk.ErrLlmTokenizationContextLength} {
		if _, failed := generationFailed(err); failed {
			t.Errorf("%v should end the response as incomplete, not fail it", err)
		}
	}
	for _, tc := range []struct {
		err    error
		status int
		code   string
		param  any
	}{
		{geniex_sdk.ErrLlmGenerationPromptTooLong, http.StatusBadRequest, "context_length_exceeded", "input"},
		{fmt.Errorf("wrapped: %w", geniex_sdk.ErrCommonInvalidInput), http.StatusBadRequest, errorCode(geniex_sdk.ErrCommonInvalidInput), nil},
		{geniex_sdk.ErrVlmImageLoad, http.StatusBadRequest, errorCode(geniex_sdk.ErrVlmImageLoad), nil},
		{fmt.Errorf("model crashed"), http.StatusInternalServerError, errorCode(fmt.Errorf("model crashed")), nil},
	} {
		f, failed := generationFailed(tc.err)
		if !failed || f.status != tc.status || f.code != tc.code || f.param != tc.param || f.message == "" {
			t.Errorf("generationFailed(%v) = %+v, %v; want status %d code %q param %v", tc.err, f, failed, tc.status, tc.code, tc.param)
		}
	}
}
