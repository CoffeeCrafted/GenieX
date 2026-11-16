// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import "encoding/json"

// The Open Responses wire types (https://www.openresponses.org/specification,
// version 2026-04-24) that POST /v1/responses reads and writes. Request unions
// stay json.RawMessage and are decoded in responses_translate.go; response
// types carry every field the spec's ResponseResource marks required.

// responsesRequest is CreateResponseBody plus the GenieX model-load and
// sampling extensions that /v1/chat/completions also takes.
type responsesRequest struct {
	Model              *string           `json:"model"`
	Input              json.RawMessage   `json:"input"`
	Instructions       *string           `json:"instructions"`
	PreviousResponseID *string           `json:"previous_response_id"`
	Tools              []responsesTool   `json:"tools"`
	ToolChoice         json.RawMessage   `json:"tool_choice"`
	ParallelToolCalls  *bool             `json:"parallel_tool_calls"`
	Text               *responsesText    `json:"text"`
	Reasoning          *responsesReason  `json:"reasoning"`
	MaxOutputTokens    *int64            `json:"max_output_tokens"`
	MaxToolCalls       *int64            `json:"max_tool_calls"`
	Temperature        *float64          `json:"temperature"`
	TopP               *float64          `json:"top_p"`
	PresencePenalty    *float64          `json:"presence_penalty"`
	FrequencyPenalty   *float64          `json:"frequency_penalty"`
	TopLogprobs        *int64            `json:"top_logprobs"`
	Stream             bool              `json:"stream"`
	Store              *bool             `json:"store"`
	Truncation         *string           `json:"truncation"`
	Metadata           map[string]string `json:"metadata"`
	Background         bool              `json:"background"`
	ServiceTier        *string           `json:"service_tier"`
	SafetyIdentifier   *string           `json:"safety_identifier"`
	PromptCacheKey     *string           `json:"prompt_cache_key"`
	Include            []string          `json:"include"`

	// GenieX extensions (the spec allows extra fields that do not change the
	// standard ones), named and defaulted as on /v1/chat/completions.
	NCtx      *int32   `json:"nctx"`
	Ngl       *int32   `json:"ngl"`
	Compute   *string  `json:"compute"`
	PowerMode *string  `json:"power_mode"`
	Seed      *int64   `json:"seed"`
	TopK      *int32   `json:"top_k"`
	MinP      *float32 `json:"min_p"`
}

// responsesItem is any input item (ItemParam), decoded by its type. Output
// items are stored in this shape too, so a later previous_response_id or
// item_reference replays them as input.
type responsesItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	Phase     string          `json:"phase,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    json.RawMessage `json:"output,omitempty"`
	// A compaction item (from POST /v1/responses/compact) carries its window here.
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

// responsesPart is one content part of a message or a function_call_output.
type responsesPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Refusal  string `json:"refusal"`
	ImageURL string `json:"image_url"`
	FileID   string `json:"file_id"`
}

// responsesTool is a FunctionToolParam; the spec defines no other tool type.
type responsesTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description *string        `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict"`
}

type responsesText struct {
	Format    json.RawMessage `json:"format,omitempty"`
	Verbosity string          `json:"verbosity,omitempty"`
}

type responsesReason struct {
	Effort  *string `json:"effort"`
	Summary *string `json:"summary"`
}

// responseResource is the ResponseResource object, returned by a blocking
// request and carried by the response.* lifecycle events.
type responseResource struct {
	ID                 string            `json:"id"`
	Object             string            `json:"object"`
	CreatedAt          int64             `json:"created_at"`
	CompletedAt        *int64            `json:"completed_at"`
	Status             string            `json:"status"`
	IncompleteDetails  *incompleteReason `json:"incomplete_details"`
	Model              string            `json:"model"`
	PreviousResponseID *string           `json:"previous_response_id"`
	Instructions       *string           `json:"instructions"`
	Output             []any             `json:"output"`
	Error              *responseError    `json:"error"`
	Tools              []functionTool    `json:"tools"`
	ToolChoice         json.RawMessage   `json:"tool_choice"`
	Truncation         string            `json:"truncation"`
	ParallelToolCalls  bool              `json:"parallel_tool_calls"`
	Text               responsesText     `json:"text"`
	TopP               float64           `json:"top_p"`
	PresencePenalty    float64           `json:"presence_penalty"`
	FrequencyPenalty   float64           `json:"frequency_penalty"`
	TopLogprobs        int64             `json:"top_logprobs"`
	Temperature        float64           `json:"temperature"`
	Reasoning          responsesReason   `json:"reasoning"`
	Usage              *responsesUsage   `json:"usage"`
	MaxOutputTokens    *int64            `json:"max_output_tokens"`
	MaxToolCalls       *int64            `json:"max_tool_calls"`
	Store              bool              `json:"store"`
	Background         bool              `json:"background"`
	ServiceTier        string            `json:"service_tier"`
	Metadata           map[string]string `json:"metadata"`
	SafetyIdentifier   *string           `json:"safety_identifier"`
	PromptCacheKey     *string           `json:"prompt_cache_key"`
	// GenieX extension: llama-server style prefill/decode timings, as chat has.
	Timings *timings `json:"timings,omitempty"`
}

type incompleteReason struct {
	Reason string `json:"reason"`
}

type responseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type functionTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description *string        `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict"`
}

type responsesUsage struct {
	InputTokens         int64              `json:"input_tokens"`
	OutputTokens        int64              `json:"output_tokens"`
	TotalTokens         int64              `json:"total_tokens"`
	InputTokensDetails  inputTokenDetails  `json:"input_tokens_details"`
	OutputTokensDetails outputTokenDetails `json:"output_tokens_details"`
}

type inputTokenDetails struct {
	CachedTokens int64 `json:"cached_tokens"`
}

type outputTokenDetails struct {
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

// Output items. Status is in_progress while streaming, then completed, or
// incomplete when generation stopped at the token limit.

type messageItem struct {
	Type    string       `json:"type"`
	ID      string       `json:"id"`
	Status  string       `json:"status"`
	Role    string       `json:"role"`
	Content []outputText `json:"content"`
}

type outputText struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
}

type reasoningItem struct {
	Type    string          `json:"type"`
	ID      string          `json:"id"`
	Summary []any           `json:"summary"`
	Content []reasoningText `json:"content"`
}

type reasoningText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type functionCallItem struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status"`
}
