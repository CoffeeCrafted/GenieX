// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

// stream writes a Responses request as Open Responses streaming events:
//
//	response.created, response.in_progress, the output item events
//	(outputBuilder), then one terminal event: response.completed or
//	response.incomplete; on a generation error, an error event and
//	response.failed (HTTP), or one error envelope (WebSocket: the client ends
//	the turn at the error, so a response.failed would land in its next turn).
//
// Every event carries a sequence_number counting from 0. Over HTTP each is an
// SSE event named by its type, and no [DONE] line follows: the spec makes it
// optional and clients must accept a stream that ends at the terminal event.
func (r *responsesRun) stream(c *gin.Context, parseTool bool, tools string, generate func(consume func(<-chan string, func() error, *geniex_sdk.ProfileData))) {
	send := r.t.events
	if send == nil {
		send = sseEvents(c)
	}
	seq := 0
	write := func(eventType string, fields map[string]any) {
		fields["type"] = eventType
		fields["sequence_number"] = seq
		seq++
		send(eventType, fields)
	}

	out := newOutputBuilder(write, r.plan.allowed, r.req.MaxToolCalls)
	write("response.created", map[string]any{"response": r.resource("in_progress", out, nil)})
	write("response.in_progress", map[string]any{"response": r.resource("in_progress", out, nil)})
	feed := r.feeder(out, parseTool, tools)

	generate(func(tokens <-chan string, wait func() error, profile *geniex_sdk.ProfileData) {
		for token := range tokens {
			if r.t.ctx.Err() != nil {
				return // client gone: generateStreaming stops the model
			}
			feed.token(token)
		}
		err := wait()
		if err != nil && !errors.Is(err, geniex_sdk.ErrLlmTokenizationContextLength) {
			status, code, message := http.StatusInternalServerError, errorCode(err), err.Error()
			if errors.Is(err, geniex_sdk.ErrLlmGenerationPromptTooLong) {
				status, code, message = http.StatusBadRequest, "context_length_exceeded", "the input is longer than the model's context window"
			}
			slog.Error("Generation error", "error", err)
			if r.t.websocket {
				r.t.fail(status, message, code, nil)
				return
			}
			write("error", map[string]any{"error": responsesErrorPayload(status, message, code, nil)})
			failed := r.resource("failed", out, nil)
			failed.Error = &responseError{Code: code, Message: message}
			write("response.failed", map[string]any{"response": failed})
			return
		}
		logProfile(*profile)
		feed.end()
		status := r.finishStatus(*profile, err)
		out.finish(itemStatus(status))
		res := r.resource(status, out, profile)
		r.store(out)
		write("response."+status, map[string]any{"response": res})
	})
}

// sseEvents starts an SSE response on c and returns a writer for its events.
func sseEvents(c *gin.Context) func(eventType string, fields map[string]any) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Status(http.StatusOK)
	return func(eventType string, fields map[string]any) {
		data, err := json.Marshal(fields)
		if err != nil {
			slog.Error("Encoding a Responses event", "type", eventType, "error", err)
			return
		}
		if _, err := fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", eventType, data); err != nil {
			slog.Debug("Writing a Responses event", "type", eventType, "error", err)
			return
		}
		c.Writer.Flush()
	}
}
