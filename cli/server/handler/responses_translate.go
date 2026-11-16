// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3"
)

// A Responses request is run as the Chat Completions request it amounts to:
// planResponses writes that request as chat JSON and decodes it with the same
// binding /v1/chat/completions uses, so templating, tool parsing, thinking and
// images behave exactly as they do there. (Building openai-go message values
// directly is not equivalent: its constructors leave the role empty.)

// responsesPlan is a validated Responses request and the chat request it runs.
type responsesPlan struct {
	param ChatCompletionRequest
	// conversation is the full input after previous_response_id and
	// item_reference are resolved; the store keeps it with the output.
	conversation []responsesItem
	// allowed limits which functions may be called (tool_choice); nil allows all.
	allowed map[string]bool
	tools   []functionTool
	choice  json.RawMessage
}

// requestError is a 4xx answer with the spec's code and param.
type requestError struct {
	status  int
	message string
	code    string
	param   string
}

func badRequest(param, format string, args ...any) *requestError {
	return &requestError{status: http.StatusBadRequest, message: fmt.Sprintf(format, args...), code: "invalid_request", param: param}
}

func unsupported(param, format string, args ...any) *requestError {
	return &requestError{status: http.StatusBadRequest, message: fmt.Sprintf(format, args...), code: "unsupported_parameter", param: param}
}

// responseLookup finds stored responses and items: the server-wide store, or
// on a WebSocket the connection's own state in front of it.
type responseLookup interface {
	get(id string) ([]responsesItem, bool)
	item(id string) (responsesItem, bool)
}

// planResponses validates req and translates it. store resolves
// previous_response_id and item_reference.
func planResponses(req responsesRequest, store responseLookup) (*responsesPlan, *requestError) {
	if req.Background {
		return nil, unsupported("background", "background responses are not supported")
	}
	if req.Truncation != nil && *req.Truncation != "auto" && *req.Truncation != "disabled" {
		return nil, badRequest("truncation", "truncation must be auto or disabled")
	}
	if rerr := checkTextFormat(req.Text); rerr != nil {
		return nil, rerr
	}

	items, rerr := parseInput(req.Input)
	if rerr != nil {
		return nil, rerr
	}
	var history []responsesItem
	if req.PreviousResponseID != nil && *req.PreviousResponseID != "" {
		prior, ok := store.get(*req.PreviousResponseID)
		if !ok {
			// 400, as in the spec's own example of this error.
			return nil, &requestError{status: http.StatusBadRequest, message: fmt.Sprintf("previous response %q was not found", *req.PreviousResponseID), code: "previous_response_not_found", param: "previous_response_id"}
		}
		history = prior
	}
	conversation := append(append([]responsesItem{}, history...), items...)
	conversation, rerr = resolveReferences(conversation, store)
	if rerr != nil {
		return nil, rerr
	}

	messages, rerr := chatMessages(req.Instructions, conversation)
	if rerr != nil {
		return nil, rerr
	}
	if !hasNonSystemMessage(messages) {
		return nil, badRequest("input", "input must contain at least one message that is not a system or developer message")
	}

	tools, allowed, choice, rerr := planTools(req.Tools, req.ToolChoice)
	if rerr != nil {
		return nil, rerr
	}

	param, rerr := chatRequest(req, messages, tools)
	if rerr != nil {
		return nil, rerr
	}
	return &responsesPlan{param: param, conversation: conversation, allowed: allowed, tools: echoTools(req.Tools), choice: choice}, nil
}

// checkTextFormat accepts plain text only: GenieX has no JSON-schema-to-
// grammar conversion, and ignoring a requested schema would hand the client
// unconstrained text it believes was constrained.
func checkTextFormat(text *responsesText) *requestError {
	if text == nil || len(text.Format) == 0 || string(text.Format) == "null" {
		return nil
	}
	var format struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(text.Format, &format); err != nil {
		return badRequest("text.format", "text.format: %v", err)
	}
	if format.Type != "text" {
		return unsupported("text.format", "text.format %q is not supported; only text output is available", format.Type)
	}
	return nil
}

// parseInput reads input as a string (one user message) or a list of items.
func parseInput(raw json.RawMessage) ([]responsesItem, *requestError) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, nil
	}
	if trimmed[0] == '"' {
		return []responsesItem{{Type: "message", Role: "user", Content: raw}}, nil
	}
	var items []responsesItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, badRequest("input", "input must be a string or a list of items: %v", err)
	}
	for i := range items {
		// The type is optional on a message and on an item reference.
		if items[i].Type == "" && items[i].Role != "" {
			items[i].Type = "message"
		}
		if items[i].Type == "" && items[i].ID != "" {
			items[i].Type = "item_reference"
		}
	}
	return items, nil
}

// resolveReferences replaces each item_reference with the stored item it names.
func resolveReferences(items []responsesItem, store responseLookup) ([]responsesItem, *requestError) {
	out := make([]responsesItem, 0, len(items))
	for _, item := range items {
		if item.Type != "item_reference" {
			out = append(out, item)
			continue
		}
		found, ok := store.item(item.ID)
		if !ok {
			return nil, badRequest("input", "item %q was not found", item.ID)
		}
		out = append(out, found)
	}
	return out, nil
}

// parseParts reads message or tool-output content: a string or a list of parts.
func parseParts(raw json.RawMessage) ([]responsesPart, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, nil
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		return []responsesPart{{Type: "input_text", Text: text}}, nil
	}
	var parts []responsesPart
	err := json.Unmarshal(raw, &parts)
	return parts, err
}

// partsText joins the text of text parts and refuses any other kind.
func partsText(parts []responsesPart, where string) (string, *requestError) {
	var b strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text", "summary_text", "reasoning_text":
			b.WriteString(p.Text)
		case "refusal":
			b.WriteString(p.Refusal)
		default:
			return "", unsupported("input", "%s: content part %q is not supported here", where, p.Type)
		}
	}
	return b.String(), nil
}

// chatMessages turns instructions and items into chat messages (as JSON
// objects). Consecutive function_call items become one assistant message with
// several tool calls, the shape chat templates expect.
func chatMessages(instructions *string, items []responsesItem) ([]map[string]any, *requestError) {
	var messages []map[string]any
	if instructions != nil && *instructions != "" {
		messages = append(messages, map[string]any{"role": "system", "content": *instructions})
	}
	var calls []map[string]any  // pending tool calls of the current assistant turn
	called := map[string]bool{} // call_ids of the function calls so far
	flush := func() {
		if len(calls) > 0 {
			messages = append(messages, map[string]any{"role": "assistant", "tool_calls": calls})
			calls = nil
		}
	}
	for i, item := range items {
		where := fmt.Sprintf("input[%d]", i)
		switch item.Type {
		case "function_call":
			if item.CallID == "" || item.Name == "" {
				return nil, badRequest("input", "%s: function_call needs call_id and name", where)
			}
			called[item.CallID] = true
			calls = append(calls, map[string]any{
				"id":       item.CallID,
				"type":     "function",
				"function": map[string]any{"name": item.Name, "arguments": item.Arguments},
			})
			continue
		case "reasoning":
			// Chat templates drop earlier turns' thinking, so it is not replayed.
			continue
		}
		if item.Type == "compaction" {
			flush()
			summary, err := decodeCompaction(item.EncryptedContent)
			if err != nil {
				return nil, badRequest("input", "%s: %v", where, err)
			}
			messages = append(messages, map[string]any{"role": "system", "content": "Summary of the conversation so far:\n" + summary})
			continue
		}
		flush()
		switch item.Type {
		case "message":
			msg, rerr := chatMessage(item, where)
			if rerr != nil {
				return nil, rerr
			}
			messages = append(messages, msg)
		case "function_call_output":
			parts, err := parseParts(item.Output)
			if err != nil {
				return nil, badRequest("input", "%s: output: %v", where, err)
			}
			text, rerr := partsText(parts, where)
			if rerr != nil {
				return nil, rerr
			}
			if item.CallID == "" {
				return nil, badRequest("input", "%s: function_call_output needs call_id", where)
			}
			// A chat template renders a tool result only against the call it
			// answers, so an output with no earlier call cannot be run.
			if !called[item.CallID] {
				return nil, badRequest("input", "%s: no function_call with call_id %q comes before this output", where, item.CallID)
			}
			messages = append(messages, map[string]any{"role": "tool", "tool_call_id": item.CallID, "content": text})
		case "":
			return nil, badRequest("input", "%s: item has no type", where)
		default:
			return nil, unsupported("input", "%s: item type %q is not supported", where, item.Type)
		}
	}
	flush()
	return messages, nil
}

func chatMessage(item responsesItem, where string) (map[string]any, *requestError) {
	parts, err := parseParts(item.Content)
	if err != nil {
		return nil, badRequest("input", "%s: content: %v", where, err)
	}
	switch item.Role {
	case "system", "developer":
		// GenieX chat templates have no developer role; it carries the same weight.
		text, rerr := partsText(parts, where)
		return map[string]any{"role": "system", "content": text}, rerr
	case "assistant":
		text, rerr := partsText(parts, where)
		return map[string]any{"role": "assistant", "content": text}, rerr
	case "user":
		return userMessage(parts, where)
	default:
		return nil, badRequest("input", "%s: role %q is not one of user, system, developer, assistant", where, item.Role)
	}
}

// userMessage keeps a text-only message a plain string (the LLM path takes it
// as is) and passes images as chat image_url parts (the VLM path saves them).
func userMessage(parts []responsesPart, where string) (map[string]any, *requestError) {
	var content []map[string]any
	textOnly := true
	for _, p := range parts {
		switch p.Type {
		case "input_text":
			content = append(content, map[string]any{"type": "text", "text": p.Text})
		case "input_image":
			if p.ImageURL == "" {
				return nil, unsupported("input", "%s: input_image needs image_url; file_id is not supported", where)
			}
			textOnly = false
			content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": p.ImageURL}})
		default:
			return nil, unsupported("input", "%s: content part %q is not supported", where, p.Type)
		}
	}
	if textOnly {
		var b strings.Builder
		for _, part := range content {
			b.WriteString(part["text"].(string))
		}
		return map[string]any{"role": "user", "content": b.String()}, nil
	}
	return map[string]any{"role": "user", "content": content}, nil
}

func hasNonSystemMessage(messages []map[string]any) bool {
	for _, m := range messages {
		if m["role"] != "system" {
			return true
		}
	}
	return false
}

// planTools applies tool_choice. "none" offers no tools; a named function or
// allowed_tools offers only those, and the same set filters the output, since
// the spec makes allowed_tools a hard constraint. "required" cannot be forced
// on a local model and is treated as "auto". choice is echoed in the response.
func planTools(tools []responsesTool, rawChoice json.RawMessage) (chat []map[string]any, allowed map[string]bool, choice json.RawMessage, rerr *requestError) {
	known := map[string]bool{}
	for i, t := range tools {
		if t.Type != "function" {
			return nil, nil, nil, unsupported("tools", "tools[%d]: tool type %q is not supported; only function tools are", i, t.Type)
		}
		if t.Name == "" {
			return nil, nil, nil, badRequest("tools", "tools[%d]: name is required", i)
		}
		known[t.Name] = true
	}

	choice = json.RawMessage(`"auto"`)
	trimmed := strings.TrimSpace(string(rawChoice))
	if trimmed != "" && trimmed != "null" {
		choice = rawChoice
		var mode string
		if json.Unmarshal(rawChoice, &mode) == nil {
			switch mode {
			case "none":
				return nil, map[string]bool{}, choice, nil
			case "auto", "required":
			default:
				return nil, nil, nil, badRequest("tool_choice", "tool_choice %q is not one of none, auto, required", mode)
			}
		} else {
			var obj struct {
				Type  string `json:"type"`
				Name  string `json:"name"`
				Mode  string `json:"mode"`
				Tools []struct {
					Type string `json:"type"`
					Name string `json:"name"`
				} `json:"tools"`
			}
			if err := json.Unmarshal(rawChoice, &obj); err != nil {
				return nil, nil, nil, badRequest("tool_choice", "tool_choice: %v", err)
			}
			allowed = map[string]bool{}
			switch obj.Type {
			case "function":
				allowed[obj.Name] = true
			case "allowed_tools":
				if obj.Mode == "none" {
					return nil, map[string]bool{}, choice, nil
				}
				for _, t := range obj.Tools {
					allowed[t.Name] = true
				}
			default:
				return nil, nil, nil, badRequest("tool_choice", "tool_choice type %q is not one of function, allowed_tools", obj.Type)
			}
			for name := range allowed {
				if !known[name] {
					return nil, nil, nil, badRequest("tool_choice", "tool_choice names %q, which is not in tools", name)
				}
			}
		}
	}

	for _, t := range tools {
		if allowed != nil && !allowed[t.Name] {
			continue
		}
		fn := map[string]any{"name": t.Name}
		if t.Description != nil {
			fn["description"] = *t.Description
		}
		if t.Parameters != nil {
			fn["parameters"] = t.Parameters
		}
		if t.Strict != nil {
			fn["strict"] = *t.Strict
		}
		chat = append(chat, map[string]any{"type": "function", "function": fn})
	}
	return chat, allowed, choice, nil
}

func echoTools(tools []responsesTool) []functionTool {
	out := make([]functionTool, 0, len(tools))
	for _, t := range tools {
		out = append(out, functionTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: t.Parameters, Strict: t.Strict})
	}
	return out
}

// chatRequest writes the chat request JSON and binds it over the server
// defaults, as ChatCompletions binds a request body.
func chatRequest(req responsesRequest, messages []map[string]any, tools []map[string]any) (ChatCompletionRequest, *requestError) {
	body := map[string]any{"messages": messages}
	if req.Model != nil && *req.Model != "" {
		body["model"] = *req.Model
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	set := func(key string, v any, ok bool) {
		if ok {
			body[key] = v
		}
	}
	set("max_completion_tokens", deref(req.MaxOutputTokens), req.MaxOutputTokens != nil)
	set("temperature", deref(req.Temperature), req.Temperature != nil)
	set("top_p", deref(req.TopP), req.TopP != nil)
	set("presence_penalty", deref(req.PresencePenalty), req.PresencePenalty != nil)
	set("frequency_penalty", deref(req.FrequencyPenalty), req.FrequencyPenalty != nil)
	set("parallel_tool_calls", deref(req.ParallelToolCalls), req.ParallelToolCalls != nil)
	set("seed", deref(req.Seed), req.Seed != nil)
	set("nctx", deref(req.NCtx), req.NCtx != nil)
	set("ngl", deref(req.Ngl), req.Ngl != nil)
	set("compute", deref(req.Compute), req.Compute != nil)
	set("power_mode", deref(req.PowerMode), req.PowerMode != nil)
	set("top_k", deref(req.TopK), req.TopK != nil)
	set("min_p", deref(req.MinP), req.MinP != nil)
	// Thinking follows reasoning.effort: "none" turns it off, any other effort
	// turns it on, and no effort keeps the server's --think default.
	if req.Reasoning != nil && req.Reasoning.Effort != nil {
		body["enable_think"] = *req.Reasoning.Effort != "none"
	}
	// Reasoning is always separated, so it becomes reasoning output items.
	body["reasoning_format"] = "deepseek"

	raw, err := json.Marshal(body)
	if err != nil {
		return ChatCompletionRequest{}, badRequest("", "encoding the request: %v", err)
	}
	param := defaultChatCompletionRequest()
	if err := json.Unmarshal(raw, &param); err != nil {
		return ChatCompletionRequest{}, badRequest("", "%v", err)
	}
	param.MaxCompletionTokens = openai.Int(param.MaxCompletionTokens.Or(param.MaxTokens.Value))
	return param, nil
}

func deref[T any](p *T) T {
	if p == nil {
		var zero T
		return zero
	}
	return *p
}
