// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"strings"
)

// outputBuilder turns classified tokens (thinking, content, completed tool
// calls) into Open Responses output items. A blocking request reads items()
// at the end; a streaming one also passes emit, which receives each item
// event in the order the spec requires:
//
//	output_item.added, content_part.added, <kind>.delta..., <kind>.done,
//	content_part.done, output_item.done
//
// for a message (output_text) or reasoning (reasoning_text) item, and
// output_item.added, function_call_arguments.delta, .done, output_item.done
// for a function call. A change of kind closes the open item.
type outputBuilder struct {
	emit  func(eventType string, fields map[string]any) // nil when not streaming
	items []any                                         // closed items, in output order

	// The open item, if any.
	kind  string // "", "message", "reasoning"
	id    string
	index int
	text  strings.Builder

	// Content that is only whitespace waits here: Gemma and others put a
	// newline between thinking and the answer or a tool call, which would
	// otherwise open an empty-looking message item.
	pendingSpace string

	allowed   map[string]bool // nil: every function may be called
	maxCalls  int64           // 0: no limit
	calls     int64
	reasoning int64 // thinking tokens seen, reported as reasoning_tokens
	stored    []responsesItem
}

func newOutputBuilder(emit func(string, map[string]any), allowed map[string]bool, maxCalls *int64) *outputBuilder {
	b := &outputBuilder{emit: emit, allowed: allowed}
	if maxCalls != nil {
		b.maxCalls = *maxCalls
	}
	return b
}

func newItemID(prefix string) string {
	return fmt.Sprintf("%s_%016x", prefix, rand.Uint64())
}

func (b *outputBuilder) send(eventType string, fields map[string]any) {
	if b.emit != nil {
		b.emit(eventType, fields)
	}
}

// thinking appends reasoning text, opening a reasoning item if needed.
func (b *outputBuilder) thinking(text string) {
	b.reasoning++
	if text == "" {
		return
	}
	if b.kind != "reasoning" {
		b.close("completed")
		b.open("reasoning")
	}
	b.text.WriteString(text)
	b.send("response.reasoning.delta", b.partFields(map[string]any{"delta": text}))
}

// content appends answer text, opening a message item once it is more than
// whitespace.
func (b *outputBuilder) content(text string) {
	if text == "" {
		return
	}
	if b.kind != "message" {
		if strings.TrimSpace(b.pendingSpace+text) == "" {
			b.pendingSpace += text
			return
		}
		b.close("completed")
		b.open("message")
		text = strings.TrimLeft(b.pendingSpace+text, " \t\r\n")
		b.pendingSpace = ""
	}
	b.text.WriteString(text)
	b.send("response.output_text.delta", b.partFields(map[string]any{"delta": text, "logprobs": []any{}}))
}

// toolCall adds a complete function call. Calls to a function tool_choice
// does not allow, and calls past max_tool_calls, are dropped.
func (b *outputBuilder) toolCall(name, arguments string) {
	b.pendingSpace = ""
	if b.allowed != nil && !b.allowed[name] {
		slog.Warn("Dropping a call to a function tool_choice does not allow", "name", name)
		return
	}
	if b.maxCalls > 0 && b.calls >= b.maxCalls {
		slog.Warn("Dropping a function call past max_tool_calls", "name", name, "max_tool_calls", b.maxCalls)
		return
	}
	b.calls++
	b.close("completed")
	id, callID, index := newItemID("fc"), newItemID("call"), len(b.items)
	b.send("response.output_item.added", map[string]any{"output_index": index, "item": functionCallItem{
		Type: "function_call", ID: id, CallID: callID, Name: name, Arguments: "", Status: "in_progress",
	}})
	b.send("response.function_call_arguments.delta", map[string]any{"item_id": id, "output_index": index, "delta": arguments})
	b.send("response.function_call_arguments.done", map[string]any{"item_id": id, "output_index": index, "arguments": arguments})
	item := functionCallItem{Type: "function_call", ID: id, CallID: callID, Name: name, Arguments: arguments, Status: "completed"}
	b.send("response.output_item.done", map[string]any{"output_index": index, "item": item})
	b.items = append(b.items, item)
	b.stored = append(b.stored, responsesItem{Type: "function_call", ID: id, CallID: callID, Name: name, Arguments: arguments})
}

func (b *outputBuilder) open(kind string) {
	b.kind, b.index = kind, len(b.items)
	b.text.Reset()
	switch kind {
	case "message":
		b.id = newItemID("msg")
		b.send("response.output_item.added", map[string]any{"output_index": b.index, "item": messageItem{
			Type: "message", ID: b.id, Status: "in_progress", Role: "assistant", Content: []outputText{},
		}})
		b.send("response.content_part.added", b.partFields(map[string]any{"part": newOutputText("")}))
	case "reasoning":
		b.id = newItemID("rs")
		b.send("response.output_item.added", map[string]any{"output_index": b.index, "item": reasoningItem{
			Type: "reasoning", ID: b.id, Summary: []any{}, Content: []reasoningText{},
		}})
		b.send("response.content_part.added", b.partFields(map[string]any{"part": reasoningText{Type: "reasoning_text", Text: ""}}))
	}
}

// close finishes the open item with status (completed, or incomplete when
// generation stopped at the token limit).
func (b *outputBuilder) close(status string) {
	text := b.text.String()
	switch b.kind {
	case "message":
		part := newOutputText(text)
		b.send("response.output_text.done", b.partFields(map[string]any{"text": text, "logprobs": []any{}}))
		b.send("response.content_part.done", b.partFields(map[string]any{"part": part}))
		item := messageItem{Type: "message", ID: b.id, Status: status, Role: "assistant", Content: []outputText{part}}
		b.send("response.output_item.done", map[string]any{"output_index": b.index, "item": item})
		b.items = append(b.items, item)
		content, _ := json.Marshal(text)
		b.stored = append(b.stored, responsesItem{Type: "message", ID: b.id, Role: "assistant", Content: content})
	case "reasoning":
		part := reasoningText{Type: "reasoning_text", Text: text}
		b.send("response.reasoning.done", b.partFields(map[string]any{"text": text}))
		b.send("response.content_part.done", b.partFields(map[string]any{"part": part}))
		item := reasoningItem{Type: "reasoning", ID: b.id, Summary: []any{}, Content: []reasoningText{part}}
		b.send("response.output_item.done", map[string]any{"output_index": b.index, "item": item})
		b.items = append(b.items, item)
	}
	b.kind = ""
}

// finish closes the open item; status is completed or incomplete.
func (b *outputBuilder) finish(status string) {
	b.close(status)
	b.pendingSpace = ""
}

func (b *outputBuilder) partFields(fields map[string]any) map[string]any {
	fields["item_id"] = b.id
	fields["output_index"] = b.index
	fields["content_index"] = 0
	return fields
}

func newOutputText(text string) outputText {
	return outputText{Type: "output_text", Text: text, Annotations: []any{}, Logprobs: []any{}}
}
