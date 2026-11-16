// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/thinkfsm"
	"github.com/qualcomm/GenieX/cli/server/types"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// POST /v1/responses/compact: the spec's standalone compaction. It returns a
// compacted input window, not a response: one compaction item a client
// passes back as the first input of a new chain (without
// previous_response_id). The window is a summary the model writes of the
// conversation (input, previous_response_id and instructions resolved as for
// /v1/responses). Its encrypted_content is that summary, base64-encoded: the
// spec makes the field opaque to clients, and GenieX does not encrypt it.

// compactRequest is CompactResponseMethodPublicBody.
type compactRequest struct {
	Model              *string         `json:"model"`
	Input              json.RawMessage `json:"input"`
	PreviousResponseID *string         `json:"previous_response_id"`
	Instructions       *string         `json:"instructions"`
	PromptCacheKey     *string         `json:"prompt_cache_key"`
}

// compactResource is CompactResource.
type compactResource struct {
	ID        string           `json:"id"`
	Object    string           `json:"object"`
	Output    []compactionItem `json:"output"`
	CreatedAt int64            `json:"created_at"`
	Usage     responsesUsage   `json:"usage"`
}

// compactionItem is CompactionBody, the one item a compaction returns.
type compactionItem struct {
	Type             string `json:"type"`
	ID               string `json:"id"`
	EncryptedContent string `json:"encrypted_content"`
	CreatedBy        string `json:"created_by"`
}

// compactionVersion marks the encoding of encrypted_content, so a later
// GenieX can change it and still read windows clients kept.
const compactionVersion = 1

type compactionContent struct {
	Version int    `json:"v"`
	Summary string `json:"summary"`
}

// compactMaxTokens bounds the summary; it replaces a whole conversation in
// the next prompt, so it has to stay much shorter than one.
const compactMaxTokens = 1024

const compactInstruction = `Summarize the conversation below so that it can be continued later from the summary alone. ` +
	`Keep every fact, name, number, code word, decision, tool result and open question the conversation relies on. ` +
	`Write only the summary, as short plain prose.`

func encodeCompaction(summary string) string {
	raw, _ := json.Marshal(compactionContent{Version: compactionVersion, Summary: summary})
	return base64.StdEncoding.EncodeToString(raw)
}

// decodeCompaction reads encrypted_content back into the summary.
func decodeCompaction(encrypted string) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(encrypted)
	if err != nil {
		return "", errors.New("encrypted_content is not a GenieX compaction")
	}
	var content compactionContent
	if err := json.Unmarshal(raw, &content); err != nil || content.Version != compactionVersion {
		return "", errors.New("encrypted_content is not a GenieX compaction")
	}
	return content.Summary, nil
}

// ResponsesCompact handles POST /v1/responses/compact.
func ResponsesCompact(c *gin.Context) {
	fail := httpResponsesErrors(c)
	useResponsesErrors(c, fail)
	var req compactRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		fail(http.StatusBadRequest, err.Error(), "invalid_request", nil)
		return
	}
	param, rerr := planCompaction(req, responsesStore)
	if rerr != nil {
		fail(rerr.status, rerr.message, rerr.code, nullable(rerr.param))
		return
	}
	slog.Info("ResponsesCompact", "model", param.Model)
	dispatchChat(c, param, compactRun{})
}

// planCompaction validates req and builds the chat request that asks the
// model for the summary: the instruction, then the conversation as a
// transcript in one user message.
func planCompaction(req compactRequest, store responseLookup) (ChatCompletionRequest, *requestError) {
	if req.Model == nil || *req.Model == "" {
		return ChatCompletionRequest{}, badRequest("model", "model is required")
	}
	items, rerr := parseInput(req.Input)
	if rerr != nil {
		return ChatCompletionRequest{}, rerr
	}
	var history []responsesItem
	if req.PreviousResponseID != nil && *req.PreviousResponseID != "" {
		prior, ok := store.get(*req.PreviousResponseID)
		if !ok {
			return ChatCompletionRequest{}, &requestError{status: http.StatusBadRequest, message: fmt.Sprintf("previous response %q was not found", *req.PreviousResponseID), code: "previous_response_not_found", param: "previous_response_id"}
		}
		history = prior
	}
	conversation, rerr := resolveReferences(append(history, items...), store)
	if rerr != nil {
		return ChatCompletionRequest{}, rerr
	}
	messages, rerr := chatMessages(req.Instructions, conversation)
	if rerr != nil {
		return ChatCompletionRequest{}, rerr
	}
	if len(messages) == 0 {
		return ChatCompletionRequest{}, badRequest("input", "there is no conversation to compact")
	}

	maxTokens := int64(compactMaxTokens)
	effort := "none" // a summary needs no visible thinking, and it is faster without
	return chatRequest(responsesRequest{
		Model:           req.Model,
		MaxOutputTokens: &maxTokens,
		Reasoning:       &responsesReason{Effort: &effort},
	}, []map[string]any{
		{"role": "system", "content": compactInstruction},
		{"role": "user", "content": transcript(messages)},
	}, nil)
}

// transcript renders chat messages as plain text for the summarizer.
func transcript(messages []map[string]any) string {
	var b strings.Builder
	for _, m := range messages {
		role, _ := m["role"].(string)
		switch content := m["content"].(type) {
		case string:
			if content != "" {
				fmt.Fprintf(&b, "%s: %s\n", transcriptRole(role, m), content)
			}
		case []map[string]any:
			var parts []string
			for _, p := range content {
				if text, ok := p["text"].(string); ok {
					parts = append(parts, text)
				} else {
					parts = append(parts, "[image]")
				}
			}
			fmt.Fprintf(&b, "%s: %s\n", transcriptRole(role, m), strings.Join(parts, " "))
		}
		if calls, ok := m["tool_calls"].([]map[string]any); ok {
			for _, call := range calls {
				fn, _ := call["function"].(map[string]any)
				fmt.Fprintf(&b, "Assistant called %v(%v)\n", fn["name"], fn["arguments"])
			}
		}
	}
	return b.String()
}

func transcriptRole(role string, m map[string]any) string {
	switch role {
	case "system":
		return "System"
	case "assistant":
		return "Assistant"
	case "tool":
		return fmt.Sprintf("Tool result for %v", m["tool_call_id"])
	default:
		return "User"
	}
}

// compactRun writes the summary as a CompactResource.
type compactRun struct{}

func (compactRun) llm(c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages []geniex_sdk.LlmChatMessage, session utils.SessionKey) {
	runCompaction(c, param, modelParam, messages, session, prepareLLM)
}

func (compactRun) vlm(c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages []geniex_sdk.VlmChatMessage, session utils.SessionKey) {
	runCompaction(c, param, modelParam, messages, session, prepareVLM)
}

func runCompaction[T, M any](c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages M, session utils.SessionKey, prepare prepareFn[T, M]) {
	job, ok := startChat(c, param, modelParam, messages, session, prepare)
	if !ok {
		return
	}
	var summary strings.Builder
	class := reasoningClass() // thinking, if the model thinks anyway, is not summary
	profile, _, err := job.generate(func(token string) bool {
		if ev := class(token); ev.Kind == thinkfsm.Content {
			summary.WriteString(ev.Text)
		}
		return true
	})
	if errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
		writeResponsesError(c, http.StatusBadRequest, "the conversation is longer than the model's context window", "context_length_exceeded", "input")
		return
	}
	if err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
		writeChatExecutionError(c, err)
		return
	}
	logProfile(profile)
	usage := profile2Usage(profile)
	c.JSON(http.StatusOK, compactResource{
		ID:     newItemID("cmp"),
		Object: "response.compaction",
		Output: []compactionItem{{
			Type:             "compaction",
			ID:               newItemID("cmpi"),
			EncryptedContent: encodeCompaction(strings.TrimSpace(summary.String())),
			CreatedBy:        "geniex",
		}},
		CreatedAt: time.Now().Unix(),
		Usage: responsesUsage{
			InputTokens:        usage.PromptTokens,
			OutputTokens:       usage.CompletionTokens,
			TotalTokens:        usage.TotalTokens,
			InputTokensDetails: inputTokenDetails{CachedTokens: usage.PromptTokensDetails.CachedTokens},
		},
	})
}
