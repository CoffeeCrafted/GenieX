// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"net/http"
	"testing"
)

func plan(t *testing.T, body string, store *responseStore) (*responsesPlan, *requestError) {
	t.Helper()
	var req responsesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	if store == nil {
		store = newResponseStore(4)
	}
	return planResponses(req, store)
}

func mustPlan(t *testing.T, body string, store *responseStore) *responsesPlan {
	t.Helper()
	p, rerr := plan(t, body, store)
	if rerr != nil {
		t.Fatalf("planResponses(%s) = %+v", body, rerr)
	}
	return p
}

// roles lists each chat message as role:text, so a test reads the translation.
func roles(p *responsesPlan) []string {
	var out []string
	for _, m := range p.param.Messages {
		entry := *m.GetRole() + ":" + contentText(m)
		for _, tc := range m.GetToolCalls() {
			entry += " call(" + *tc.GetID() + "," + tc.GetFunction().Name + "," + tc.GetFunction().Arguments + ")"
		}
		if id := m.GetToolCallID(); id != nil {
			entry += " for " + *id
		}
		out = append(out, entry)
	}
	return out
}

func equal(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got %q, want %q", got, want)
		}
	}
}

func TestResponsesStringInputIsOneUserMessage(t *testing.T) {
	p := mustPlan(t, `{"input":"Say hi."}`, nil)
	equal(t, roles(p), []string{"user:Say hi."})
}

func TestResponsesMessagesKeepTheirRoles(t *testing.T) {
	p := mustPlan(t, `{
		"instructions": "Be brief.",
		"input": [
			{"type":"message","role":"developer","content":"Use metric units."},
			{"role":"user","content":[{"type":"input_text","text":"My name is "},{"type":"input_text","text":"Alice."}]},
			{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"Hello Alice."}]},
			{"type":"message","role":"user","content":"What is my name?"}
		]}`, nil)
	equal(t, roles(p), []string{
		"system:Be brief.",
		"system:Use metric units.",
		"user:My name is Alice.",
		"assistant:Hello Alice.",
		"user:What is my name?",
	})
}

func TestResponsesConsecutiveCallsShareOneAssistantTurn(t *testing.T) {
	p := mustPlan(t, `{"input": [
		{"role":"user","content":"Weather in Paris and Oslo?"},
		{"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
		{"type":"function_call","call_id":"c2","name":"get_weather","arguments":"{\"city\":\"Oslo\"}"},
		{"type":"function_call_output","call_id":"c1","output":"18C"},
		{"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":"9C"}]}
	]}`, nil)
	equal(t, roles(p), []string{
		"user:Weather in Paris and Oslo?",
		`assistant: call(c1,get_weather,{"city":"Paris"}) call(c2,get_weather,{"city":"Oslo"})`,
		"tool:18C for c1",
		"tool:9C for c2",
	})
}

func TestResponsesReasoningItemsAreNotReplayed(t *testing.T) {
	p := mustPlan(t, `{"input": [
		{"role":"user","content":"Hi"},
		{"type":"reasoning","summary":[]},
		{"role":"assistant","content":"Hello"},
		{"role":"user","content":"Again"}
	]}`, nil)
	equal(t, roles(p), []string{"user:Hi", "assistant:Hello", "user:Again"})
}

func TestResponsesImageBecomesAChatImagePart(t *testing.T) {
	p := mustPlan(t, `{"input":[{"role":"user","content":[
		{"type":"input_text","text":"What is this?"},
		{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}]}`, nil)
	raw, err := json.Marshal(p.param.Messages[0])
	if err != nil {
		t.Fatal(err)
	}
	var msg struct {
		Content []struct {
			Type     string `json:"type"`
			ImageURL struct {
				URL string `json:"url"`
			} `json:"image_url"`
		} `json:"content"`
	}
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 2 || msg.Content[1].Type != "image_url" || msg.Content[1].ImageURL.URL != "data:image/png;base64,AAAA" {
		t.Fatalf("user content = %s, want a text part and an image_url part", raw)
	}
}

func TestResponsesReasoningEffortControlsThinking(t *testing.T) {
	if p := mustPlan(t, `{"input":"x","reasoning":{"effort":"none"}}`, nil); p.param.EnableThink {
		t.Error(`effort "none" should turn thinking off`)
	}
	if p := mustPlan(t, `{"input":"x","reasoning":{"effort":"low"}}`, nil); !p.param.EnableThink {
		t.Error(`effort "low" should turn thinking on`)
	}
	if p := mustPlan(t, `{"input":"x"}`, nil); p.param.ReasoningFormat != "deepseek" {
		t.Errorf("reasoning_format = %q, want deepseek so thinking becomes reasoning items", p.param.ReasoningFormat)
	}
}

func TestResponsesSamplingAndLimits(t *testing.T) {
	p := mustPlan(t, `{"input":"x","max_output_tokens":64,"temperature":0.2,"top_p":0.9,"seed":7,"parallel_tool_calls":false}`, nil)
	if got := p.param.MaxCompletionTokens.Value; got != 64 {
		t.Errorf("max_completion_tokens = %d, want 64", got)
	}
	if p.param.Temperature.Value != 0.2 || p.param.TopP.Value != 0.9 || p.param.Seed.Value != 7 {
		t.Errorf("sampling = temperature %v, top_p %v, seed %v", p.param.Temperature.Value, p.param.TopP.Value, p.param.Seed.Value)
	}
	if p.param.ParallelToolCalls.Value {
		t.Error("parallel_tool_calls false was not passed through")
	}
	if p := mustPlan(t, `{"input":"x"}`, nil); p.param.MaxCompletionTokens.Value != 2048 {
		t.Errorf("default max tokens = %d, want the chat default 2048", p.param.MaxCompletionTokens.Value)
	}
}

const weatherTools = `"tools":[
	{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}},
	{"type":"function","name":"get_time"}]`

func toolNames(p *responsesPlan) []string {
	var names []string
	for _, tool := range p.param.Tools {
		names = append(names, tool.GetFunction().Name)
	}
	return names
}

func TestResponsesToolChoice(t *testing.T) {
	if p := mustPlan(t, `{"input":"x",`+weatherTools+`}`, nil); len(p.param.Tools) != 2 || p.allowed != nil {
		t.Errorf("auto: tools %v allowed %v, want both and no filter", toolNames(p), p.allowed)
	}
	if p := mustPlan(t, `{"input":"x",`+weatherTools+`,"tool_choice":"none"}`, nil); len(p.param.Tools) != 0 {
		t.Errorf("none: tools %v, want none offered", toolNames(p))
	}
	p := mustPlan(t, `{"input":"x",`+weatherTools+`,"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"get_time"}]}}`, nil)
	equal(t, toolNames(p), []string{"get_time"})
	if !p.allowed["get_time"] || p.allowed["get_weather"] {
		t.Errorf("allowed = %v, want only get_time", p.allowed)
	}
	p = mustPlan(t, `{"input":"x",`+weatherTools+`,"tool_choice":{"type":"function","name":"get_weather"}}`, nil)
	equal(t, toolNames(p), []string{"get_weather"})
	if string(p.choice) != `{"type":"function","name":"get_weather"}` {
		t.Errorf("echoed tool_choice = %s", p.choice)
	}
	if _, rerr := plan(t, `{"input":"x",`+weatherTools+`,"tool_choice":{"type":"function","name":"nope"}}`, nil); rerr == nil || rerr.param != "tool_choice" {
		t.Errorf("unknown function in tool_choice: %+v, want a tool_choice error", rerr)
	}
}

func TestResponsesRejectsWhatItCannotHonour(t *testing.T) {
	for name, body := range map[string]string{
		"json schema":   `{"input":"x","text":{"format":{"type":"json_schema","name":"a","schema":{}}}}`,
		"background":    `{"input":"x","background":true}`,
		"input file":    `{"input":[{"role":"user","content":[{"type":"input_file","file_data":"x"}]}]}`,
		"non-function":  `{"input":"x","tools":[{"type":"web_search"}]}`,
		"no input":      `{}`,
		"only system":   `{"input":[{"role":"system","content":"x"}]}`,
		"unknown role":  `{"input":[{"role":"robot","content":"x"}]}`,
		"call, no id":   `{"input":[{"type":"function_call","name":"f","arguments":"{}"}]}`,
		"bad truncate":  `{"input":"x","truncation":"sometimes"}`,
		"bad choice":    `{"input":"x","tool_choice":"always"}`,
		"unknown item":  `{"input":[{"type":"compaction","encrypted_content":"x"}]}`,
		"missing image": `{"input":[{"role":"user","content":[{"type":"input_image","file_id":"f1"}]}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, rerr := plan(t, body, nil); rerr == nil || rerr.status != http.StatusBadRequest {
				t.Errorf("planResponses(%s) = %+v, want a 400", body, rerr)
			}
		})
	}
	if p := mustPlan(t, `{"input":"x","text":{"format":{"type":"text"}}}`, nil); p == nil {
		t.Error("text.format text should be accepted")
	}
}

func TestResponsesPreviousResponseAndItemReference(t *testing.T) {
	store := newResponseStore(4)
	if _, rerr := plan(t, `{"input":"x","previous_response_id":"resp_missing"}`, store); rerr == nil || rerr.status != http.StatusBadRequest || rerr.code != "previous_response_not_found" {
		t.Fatalf("missing previous response: %+v, want 400 previous_response_not_found", rerr)
	}

	first := mustPlan(t, `{"input":"My name is Alice."}`, store)
	answer := responsesItem{Type: "message", ID: "msg_1", Role: "assistant", Content: json.RawMessage(`"Hello Alice."`)}
	store.put("resp_1", append(first.conversation, answer))

	p := mustPlan(t, `{"input":"What is my name?","previous_response_id":"resp_1"}`, store)
	equal(t, roles(p), []string{"user:My name is Alice.", "assistant:Hello Alice.", "user:What is my name?"})

	p = mustPlan(t, `{"input":[{"type":"item_reference","id":"msg_1"},{"role":"user","content":"Again?"}]}`, store)
	equal(t, roles(p), []string{"assistant:Hello Alice.", "user:Again?"})
	if _, rerr := plan(t, `{"input":[{"type":"item_reference","id":"msg_nope"},{"role":"user","content":"x"}]}`, store); rerr == nil {
		t.Error("an unknown item_reference should be a 400")
	}
}

func TestResponseStoreForgetsTheOldest(t *testing.T) {
	store := newResponseStore(2)
	store.put("a", []responsesItem{{Type: "message", ID: "item_a"}})
	store.put("b", []responsesItem{{Type: "message", ID: "item_a"}, {Type: "message", ID: "item_b"}})
	store.put("c", []responsesItem{{Type: "message", ID: "item_c"}})
	if _, ok := store.get("a"); ok {
		t.Error("the oldest response should be forgotten")
	}
	if _, ok := store.item("item_a"); !ok {
		t.Error("an item a kept response still holds must stay")
	}
	if _, ok := store.get("c"); !ok {
		t.Error("the newest response should be kept")
	}
}
