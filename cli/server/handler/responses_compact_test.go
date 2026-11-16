// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestCompactionContentRoundTrips(t *testing.T) {
	encoded := encodeCompaction("The code word is slate.")
	summary, err := decodeCompaction(encoded)
	if err != nil || summary != "The code word is slate." {
		t.Fatalf("decode(encode) = %q, %v", summary, err)
	}
	for _, bad := range []string{"", "not base64!", "eyJ2Ijo5OX0="} { // the last is {"v":99}
		if _, err := decodeCompaction(bad); err == nil {
			t.Errorf("decodeCompaction(%q) should fail", bad)
		}
	}
}

func planCompact(t *testing.T, body string) (ChatCompletionRequest, *requestError) {
	t.Helper()
	var req compactRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	return planCompaction(req, newResponseStore(4))
}

func TestCompactionNeedsAModel(t *testing.T) {
	if _, rerr := planCompact(t, `{"input":"Compact this."}`); rerr == nil || rerr.status != http.StatusBadRequest || rerr.param != "model" {
		t.Errorf("missing model: %+v, want a 400 on model", rerr)
	}
	if _, rerr := planCompact(t, `{"model":"m"}`); rerr == nil || rerr.param != "input" {
		t.Errorf("empty conversation: %+v, want a 400 on input", rerr)
	}
	if _, rerr := planCompact(t, `{"model":"m","input":"x","previous_response_id":"resp_nope"}`); rerr == nil || rerr.code != "previous_response_not_found" {
		t.Errorf("unknown previous response: %+v", rerr)
	}
}

func TestCompactionAsksForASummaryOfTheTranscript(t *testing.T) {
	param, rerr := planCompact(t, `{"model":"m","instructions":"Be brief.","input":[
		{"role":"user","content":[{"type":"input_text","text":"Remember the code word: slate."}]},
		{"role":"assistant","content":[{"type":"output_text","text":"OK."}]},
		{"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":"found"}
	]}`)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if len(param.Messages) != 2 || *param.Messages[0].GetRole() != "system" || *param.Messages[1].GetRole() != "user" {
		t.Fatalf("messages = %v, want the instruction then one transcript", roles(&responsesPlan{param: param}))
	}
	if !strings.Contains(contentText(param.Messages[0]), "Summarize") {
		t.Errorf("system message = %q, want the summary instruction", contentText(param.Messages[0]))
	}
	got := contentText(param.Messages[1])
	for _, want := range []string{"System: Be brief.", "User: Remember the code word: slate.", "Assistant: OK.", "Assistant called lookup({})", "Tool result for c1: found"} {
		if !strings.Contains(got, want) {
			t.Errorf("transcript lacks %q:\n%s", want, got)
		}
	}
	if param.EnableThink || param.MaxCompletionTokens.Value != compactMaxTokens {
		t.Errorf("thinking %v, max tokens %d; want off and %d", param.EnableThink, param.MaxCompletionTokens.Value, compactMaxTokens)
	}
}

func TestCompactionItemStartsANewChain(t *testing.T) {
	item, _ := json.Marshal(compactionItem{Type: "compaction", ID: "cmpi_1", EncryptedContent: encodeCompaction("The code word is slate."), CreatedBy: "geniex"})
	p := mustPlan(t, `{"input":[`+string(item)+`,{"role":"user","content":"What is the code word?"}]}`, nil)
	equal(t, roles(p), []string{
		"system:Summary of the conversation so far:\nThe code word is slate.",
		"user:What is the code word?",
	})
	if _, rerr := plan(t, `{"input":[{"type":"compaction","encrypted_content":"x"},{"role":"user","content":"y"}]}`, nil); rerr == nil || rerr.status != http.StatusBadRequest {
		t.Errorf("a compaction this server did not write: %+v, want a 400", rerr)
	}
}

func TestToolOutputMustAnswerAnEarlierCall(t *testing.T) {
	if _, rerr := plan(t, `{"input":[{"type":"function_call_output","call_id":"call_missing","output":"x"},{"role":"user","content":"y"}]}`, nil); rerr == nil || rerr.param != "input" {
		t.Errorf("unanswered output: %+v, want a 400 on input", rerr)
	}
}
