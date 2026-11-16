// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gin-gonic/gin"
)

// wsTestSession is a server-side session over a real WebSocket, with the
// client end to read what the server sends. Every turn here fails before a
// model is needed (validation, lookup), so no SDK model is loaded.
func wsTestSession(t *testing.T) (*wsSession, *websocket.Conn, *gin.Context) {
	t.Helper()
	serverConn := make(chan *websocket.Conn, 1)
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("accept: %v", err)
			return
		}
		serverConn <- conn
		<-done // keep the handler, and so the connection, alive for the test
	}))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() {
		client.CloseNow()
		close(done)
		srv.Close()
		cancel()
	})
	s := newWSSession(<-serverConn, ctx, cancel)
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	return s, client, c
}

type wsEnvelope struct {
	Type   string `json:"type"`
	Status int    `json:"status"`
	Error  struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Param   any    `json:"param"`
	} `json:"error"`
	SequenceNumber *int `json:"sequence_number"`
}

func readEnvelope(t *testing.T, client *websocket.Conn) wsEnvelope {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, data, err := client.Read(ctx)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var env wsEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		t.Fatalf("decode %s: %v", data, err)
	}
	if env.Type != "error" || env.Status == 0 || env.Error.Code == "" || env.Error.Message == "" {
		t.Fatalf("message %s is not a WebSocket error envelope", data)
	}
	if env.SequenceNumber != nil {
		t.Errorf("the error envelope has no sequence_number in the spec: %s", data)
	}
	return env
}

func TestWebSocketRejectsOtherMessagesAndHTTPOnlyFields(t *testing.T) {
	s, client, c := wsTestSession(t)
	for _, tc := range []struct{ msg, param string }{
		{`{"type":"response.cancel"}`, "type"},
		{`{"type":"response.create","input":"x","stream":true}`, "stream"},
		{`{"type":"response.create","input":"x","stream_options":{}}`, "stream_options"},
		{`{"type":"response.create","input":"x","background":false}`, "background"},
	} {
		s.turn(c, []byte(tc.msg))
		env := readEnvelope(t, client)
		if env.Status != http.StatusBadRequest || env.Error.Code != "invalid_request" || env.Error.Param != tc.param {
			t.Errorf("%s: envelope %+v, want 400 invalid_request on %s", tc.msg, env, tc.param)
		}
	}
	s.turn(c, []byte(`not json`))
	if env := readEnvelope(t, client); env.Error.Code != "invalid_request" {
		t.Errorf("bad JSON: %+v", env)
	}
}

func TestWebSocketMissingPreviousResponse(t *testing.T) {
	s, client, c := wsTestSession(t)
	s.turn(c, []byte(`{"type":"response.create","store":false,"previous_response_id":"resp_nope","input":"x"}`))
	env := readEnvelope(t, client)
	if env.Status != http.StatusBadRequest || env.Error.Code != "previous_response_not_found" || env.Error.Param != "previous_response_id" {
		t.Errorf("envelope %+v, want 400 previous_response_not_found", env)
	}
}

func TestWebSocketFailedContinuationEvictsTheResponse(t *testing.T) {
	s, client, c := wsTestSession(t)
	s.remember("resp_1", []responsesItem{
		{Type: "message", Role: "user", Content: json.RawMessage(`"Remember: ember."`)},
		{Type: "message", ID: "msg_1", Role: "assistant", Content: json.RawMessage(`"OK"`)},
	})

	// A tool output that answers no call in resp_1 fails the turn...
	s.turn(c, []byte(`{"type":"response.create","store":false,"previous_response_id":"resp_1",
		"input":[{"type":"function_call_output","call_id":"call_missing","output":"x"}]}`))
	if env := readEnvelope(t, client); env.Status != http.StatusBadRequest {
		t.Fatalf("failed continuation: %+v, want a 400", env)
	}
	if _, ok := s.get("resp_1"); ok {
		t.Fatal("a failed continuation must evict the response it continued")
	}
	// ...so continuing from it again is previous_response_not_found.
	s.turn(c, []byte(`{"type":"response.create","store":false,"previous_response_id":"resp_1","input":"stale"}`))
	if env := readEnvelope(t, client); env.Error.Code != "previous_response_not_found" {
		t.Errorf("retry after eviction: %+v, want previous_response_not_found", env)
	}
}

func TestWebSocketSessionKeepsRecentResponses(t *testing.T) {
	s := newWSSession(nil, context.Background(), func() {})
	for i := 0; i <= wsRecentResponses; i++ {
		s.remember(fmt.Sprintf("resp_%d", i), []responsesItem{{Type: "message", ID: fmt.Sprintf("msg_%d", i)}})
	}
	if _, ok := s.get("resp_0"); ok {
		t.Error("the oldest response past the limit should be forgotten")
	}
	if _, ok := s.get(fmt.Sprintf("resp_%d", wsRecentResponses)); !ok {
		t.Error("the latest response should be kept")
	}
	if _, ok := s.item(fmt.Sprintf("msg_%d", wsRecentResponses)); !ok {
		t.Error("items of kept responses are found by item_reference")
	}

	// A stored (store:true) response from anywhere is found through the global store.
	responsesStore.put("resp_global_test", []responsesItem{{Type: "message", ID: "msg_global_test"}})
	if _, ok := s.get("resp_global_test"); !ok {
		t.Error("the connection should fall back to the server-wide store")
	}
}

// TestWebSocketUpgradeThroughGin dials the real handler behind a gin engine,
// which the session tests above skip: gin v1.11 refuses a hijack once its
// writer counts as written, and that broke every upgrade.
func TestWebSocketUpgradeThroughGin(t *testing.T) {
	gin.SetMode(gin.DebugMode) // so a "headers already written" warning would show
	var log strings.Builder
	gin.DefaultWriter, gin.DefaultErrorWriter = &log, &log
	t.Cleanup(func() { gin.SetMode(gin.TestMode) })
	r := gin.New()
	r.GET("/v1/responses", ResponsesWebSocket)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	client, resp, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(srv.URL, "http")+"/v1/responses", nil)
	if err != nil {
		t.Fatalf("dial through gin: %v", err)
	}
	defer client.CloseNow()
	if resp.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("upgrade status = %d, want 101", resp.StatusCode)
	}
	if err := client.Write(ctx, websocket.MessageText, []byte(`{"type":"response.create","store":false,"previous_response_id":"resp_nope","input":"x"}`)); err != nil {
		t.Fatal(err)
	}
	if env := readEnvelope(t, client); env.Error.Code != "previous_response_not_found" {
		t.Errorf("turn over gin: %+v, want previous_response_not_found", env)
	}
	client.Close(websocket.StatusNormalClosure, "")
	time.Sleep(100 * time.Millisecond) // let the handler return and gin finish the request
	if strings.Contains(log.String(), "already written") || strings.Contains(log.String(), "hijacked") {
		t.Errorf("gin warned about the upgraded response:\n%s", log.String())
	}
}

func TestWebSocketRouteNeedsAnUpgrade(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/v1/responses", ResponsesWebSocket)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/v1/responses", nil))
	if w.Code < 400 {
		t.Errorf("a plain GET got %d, want the upgrade refused", w.Code)
	}
}

func TestWebSocketAcceptOptionsFollowOrigins(t *testing.T) {
	if opts := wsAcceptOptions("*"); !opts.InsecureSkipVerify {
		t.Error(`origins "*" should accept any origin`)
	}
	opts := wsAcceptOptions("https://app.example, http://localhost:5173")
	if opts.InsecureSkipVerify || len(opts.OriginPatterns) != 2 || opts.OriginPatterns[0] != "app.example" || opts.OriginPatterns[1] != "localhost:5173" {
		t.Errorf("origin patterns = %+v", opts)
	}
}
