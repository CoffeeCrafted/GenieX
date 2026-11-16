// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
	"github.com/qualcomm/GenieX/cli/internal/thinkfsm"
	"github.com/qualcomm/GenieX/cli/server/types"
	"github.com/qualcomm/GenieX/cli/server/utils"
)

// /v1/responses: the Open Responses create endpoint
// (https://www.openresponses.org/specification, 2026-04-24), over HTTP (POST,
// one JSON body or SSE) and over WebSocket (GET with an upgrade, one turn per
// response.create message; responses_ws.go). The request is translated to the
// chat request it amounts to (responses_translate.go) and runs through the
// chat pipeline (dispatchChat, startChat); only the output differs: output
// items (responses_output.go) in a ResponseResource, or the spec's streaming
// events (responses_stream.go). Compaction is POST /v1/responses/compact
// (responses_compact.go). Not supported: background, text formats other than
// text, and input files.

// responsesTransport is how one Responses request is delivered.
type responsesTransport struct {
	ctx context.Context
	// events sends one streaming event; nil over HTTP, where stream() writes SSE.
	events func(eventType string, fields map[string]any)
	// fail answers the request with an error (HTTP body or WebSocket envelope).
	fail errorWriter
	// lookup resolves previous_response_id and item_reference.
	lookup responseLookup
	// remember keeps each finished response in connection-local state
	// (WebSocket), whether or not it is stored; nil over HTTP.
	remember func(id string, conversation []responsesItem)
	// websocket: a failed turn is one error envelope, with no response.failed.
	websocket bool
}

// Responses handles POST /v1/responses.
func Responses(c *gin.Context) {
	fail := httpResponsesErrors(c)
	useResponsesErrors(c, fail)
	var req responsesRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		fail(http.StatusBadRequest, err.Error(), "invalid_request", nil)
		return
	}
	runResponsesRequest(c, req, responsesTransport{ctx: c.Request.Context(), fail: fail, lookup: responsesStore})
}

// runResponsesRequest plans req and runs it; errors go to t.fail.
func runResponsesRequest(c *gin.Context, req responsesRequest, t responsesTransport) {
	plan, rerr := planResponses(req, t.lookup)
	if rerr != nil {
		t.fail(rerr.status, rerr.message, rerr.code, nullable(rerr.param))
		return
	}
	slog.Info("Responses", "model", plan.param.Model, "items", len(plan.conversation), "stream", req.Stream || t.events != nil)
	run := &responsesRun{
		req:     req,
		plan:    plan,
		t:       t,
		id:      newItemID("resp"),
		created: time.Now().Unix(),
	}
	dispatchChat(c, plan.param, run)
}

// responsesRun is one Responses request inside the chat pipeline.
type responsesRun struct {
	req     responsesRequest
	plan    *responsesPlan
	t       responsesTransport
	id      string
	created int64
}

func (r *responsesRun) streaming() bool {
	return r.req.Stream || r.t.events != nil
}

func (r *responsesRun) llm(c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages []geniex_sdk.LlmChatMessage, session utils.SessionKey) {
	runResponses(c, r, param, modelParam, messages, session, prepareLLM)
}

func (r *responsesRun) vlm(c *gin.Context, param ChatCompletionRequest, modelParam types.ModelParam, messages []geniex_sdk.VlmChatMessage, session utils.SessionKey) {
	runResponses(c, r, param, modelParam, messages, session, prepareVLM)
}

func runResponses[T, M any](c *gin.Context, r *responsesRun, param ChatCompletionRequest, modelParam types.ModelParam, messages M, session utils.SessionKey, prepare prepareFn[T, M]) {
	job, ok := startChat(c, param, modelParam, messages, session, prepare)
	if !ok {
		return
	}
	if r.streaming() {
		r.stream(c, job.parseTool, job.tools, func(consume func(<-chan string, func() error, *geniex_sdk.ProfileData)) {
			generateStreaming(job, consume)
		})
		return
	}
	out := newOutputBuilder(nil, r.plan.allowed, r.req.MaxToolCalls)
	feed := r.feeder(out, job.parseTool, job.tools)
	profile, _, err := job.generate(func(token string) bool { feed.token(token); return true })
	if errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
		r.t.fail(http.StatusBadRequest, "the input is longer than the model's context window", "context_length_exceeded", "input")
		return
	}
	if err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
		writeChatExecutionError(c, err)
		return
	}
	logProfile(profile)
	feed.end()
	status := r.finishStatus(profile, err)
	out.finish(itemStatus(status))
	resource := r.resource(status, out, &profile)
	r.store(out)
	c.JSON(http.StatusOK, resource)
}

// tokenFeed classifies each token and passes it to the output builder,
// through the tool-call parser when tools were offered.
type tokenFeed struct {
	token func(string)
	end   func()
}

func (r *responsesRun) feeder(out *outputBuilder, parseTool bool, tools string) tokenFeed {
	class := reasoningClass()
	write := func(ev thinkfsm.Event) {
		switch ev.Kind {
		case thinkfsm.Thinking:
			out.thinking(ev.Text)
		case thinkfsm.Content:
			out.content(ev.Text)
		}
	}
	if !parseTool {
		return tokenFeed{token: func(t string) { write(class(t)) }, end: func() {}}
	}
	parser := utils.NewToolCallParser(utils.ToolParameterTypesFromTools(tools))
	writeAll := func(events []utils.ToolCallEvent) {
		for _, ev := range events {
			if ev.Kind == thinkfsm.ToolCall {
				out.toolCall(ev.Call.Name, ev.Call.Arguments)
				continue
			}
			write(thinkfsm.Event{Kind: ev.Kind, Text: ev.Text})
		}
	}
	return tokenFeed{
		token: func(t string) { writeAll(parser.Feed(class(t))) },
		end:   func() { writeAll(parser.Finish()) },
	}
}

// finishStatus is incomplete when generation stopped at max_output_tokens or
// ran out of context window, and completed otherwise.
func (r *responsesRun) finishStatus(profile geniex_sdk.ProfileData, err error) string {
	if profile.StopReason == "length" || errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
		return "incomplete"
	}
	return "completed"
}

func itemStatus(responseStatus string) string {
	if responseStatus == "incomplete" {
		return "incomplete"
	}
	return "completed"
}

// store keeps the conversation for previous_response_id: in the server-wide
// store unless store is false, and always in the connection's own state on
// a WebSocket, which is what lets a store:false chain continue there.
func (r *responsesRun) store(out *outputBuilder) {
	conversation := append(append([]responsesItem{}, r.plan.conversation...), out.stored...)
	if r.t.remember != nil {
		r.t.remember(r.id, conversation)
	}
	if r.req.Store != nil && !*r.req.Store {
		return
	}
	responsesStore.put(r.id, conversation)
}

// resource is the ResponseResource for status. profile is nil before
// generation ends (the created and in_progress events), so usage is null.
func (r *responsesRun) resource(status string, out *outputBuilder, profile *geniex_sdk.ProfileData) responseResource {
	req := r.req
	res := responseResource{
		ID:                 r.id,
		Object:             "response",
		CreatedAt:          r.created,
		Status:             status,
		Model:              string(r.plan.param.Model),
		PreviousResponseID: req.PreviousResponseID,
		Instructions:       req.Instructions,
		Output:             []any{},
		Tools:              r.plan.tools,
		ToolChoice:         r.plan.choice,
		Truncation:         "disabled",
		ParallelToolCalls:  r.plan.param.ParallelToolCalls.Or(true),
		Text:               responsesText{Format: json.RawMessage(`{"type":"text"}`)},
		TopP:               orDefault(req.TopP, 1),
		PresencePenalty:    orDefault(req.PresencePenalty, 0),
		FrequencyPenalty:   orDefault(req.FrequencyPenalty, 0),
		TopLogprobs:        0,
		Temperature:        orDefault(req.Temperature, 1),
		MaxOutputTokens:    req.MaxOutputTokens,
		MaxToolCalls:       req.MaxToolCalls,
		Store:              req.Store == nil || *req.Store,
		Background:         false,
		ServiceTier:        "default",
		Metadata:           req.Metadata,
		SafetyIdentifier:   req.SafetyIdentifier,
		PromptCacheKey:     req.PromptCacheKey,
	}
	if req.Truncation != nil {
		res.Truncation = *req.Truncation
	}
	if req.Reasoning != nil {
		res.Reasoning = *req.Reasoning
	}
	if res.Metadata == nil {
		res.Metadata = map[string]string{}
	}
	if status == "incomplete" {
		res.IncompleteDetails = &incompleteReason{Reason: "max_output_tokens"}
	}
	if out != nil {
		res.Output = append(res.Output, out.items...)
	}
	if profile != nil {
		done := time.Now().Unix()
		res.CompletedAt = &done
		usage := profile2Usage(*profile)
		// The SDK does not count thinking tokens; the builder counts the
		// tokens the thinking classifier saw, capped at the output total.
		var reasoningTokens int64
		if out != nil {
			reasoningTokens = min(out.reasoning, usage.CompletionTokens)
		}
		res.Usage = &responsesUsage{
			InputTokens:         usage.PromptTokens,
			OutputTokens:        usage.CompletionTokens,
			TotalTokens:         usage.TotalTokens,
			InputTokensDetails:  inputTokenDetails{CachedTokens: usage.PromptTokensDetails.CachedTokens},
			OutputTokensDetails: outputTokenDetails{ReasoningTokens: reasoningTokens},
		}
		t := profile2Timings(*profile)
		res.Timings = &t
	}
	return res
}

func orDefault(p *float64, fallback float64) float64 {
	if p == nil {
		return fallback
	}
	return *p
}

func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// errorCode is the SDK code of err as the string the spec's error.code wants.
func errorCode(err error) string {
	return fmt.Sprint(geniex_sdk.SDKErrorCode(err))
}
