// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gin-gonic/gin"

	"github.com/qualcomm/GenieX/cli/internal/config"
	"github.com/qualcomm/GenieX/cli/server/middleware"
)

// The Open Responses WebSocket transport (spec "WebSocket Transport"): GET
// /v1/responses upgrades to a WebSocket; each text message
// {"type":"response.create", ...CreateResponseBody} is one turn, answered with
// the same streaming event objects as HTTP, one JSON message per event. Turns
// run one at a time. The connection keeps its recent responses, stored or not,
// so a store:false chain continues on the same socket; a failed continuation
// evicts the response it continued; failures are one error envelope.

const (
	// wsConnectionLimit is the spec's connection lifetime.
	wsConnectionLimit = 60 * time.Minute
	// wsReadLimit allows inline images; the library's default is 32 KiB.
	wsReadLimit = 64 << 20
	// wsRecentResponses is how many responses a connection keeps; the spec
	// asks for at least the most recent.
	wsRecentResponses = 16
	// wsWriteTimeout bounds one message write, so a stalled client cannot hold
	// the GIL through a turn.
	wsWriteTimeout = 30 * time.Second
)

// ResponsesWebSocket handles GET /v1/responses. It is registered outside the
// GIL middleware: a connection lives for up to an hour, so it takes the GIL
// per turn, letting other requests run between turns.
func ResponsesWebSocket(c *gin.Context) {
	conn, err := websocket.Accept(ginUpgradeWriter{c.Writer}, c.Request, wsAcceptOptions(config.Get().Origins))
	if err != nil {
		slog.Warn("WebSocket upgrade failed", "error", err)
		return // Accept has written the HTTP error
	}
	defer conn.CloseNow()
	conn.SetReadLimit(wsReadLimit)

	ctx, cancel := context.WithTimeout(c.Request.Context(), wsConnectionLimit)
	defer cancel()
	s := &wsSession{conn: conn, ctx: ctx, cancel: cancel, responses: map[string][]responsesItem{}}
	for {
		kind, data, err := conn.Read(ctx)
		if err != nil {
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				s.sendError(http.StatusBadRequest, "the WebSocket connection reached its 60-minute limit; open a new connection", "websocket_connection_limit_reached", nil)
				conn.Close(websocket.StatusNormalClosure, "connection limit reached")
			}
			return
		}
		if kind != websocket.MessageText {
			s.sendError(http.StatusBadRequest, "messages must be JSON text", "invalid_request", nil)
			continue
		}
		s.turn(c, data)
	}
}

// ginUpgradeWriter lets coder/websocket upgrade a gin request. Accept writes
// the 101 with WriteHeader and relies on net/http to flush it when the
// connection is hijacked; for old gin versions it also calls gin's
// WriteHeaderNow, which in gin v1.11 marks the response written, and gin's
// Hijack then refuses ("gin: response already written"). So this writer has
// no WriteHeaderNow, sends the 101 to net/http's own writer, and hijacks
// through gin, which marks the response written so gin writes no status on
// the hijacked connection when the request ends. Error answers before the
// upgrade go through gin as usual.
type ginUpgradeWriter struct {
	gin gin.ResponseWriter
}

func (w ginUpgradeWriter) Header() http.Header { return w.gin.Header() }

func (w ginUpgradeWriter) Write(b []byte) (int, error) { return w.gin.Write(b) }

func (w ginUpgradeWriter) WriteHeader(code int) {
	if raw, ok := w.gin.(interface{ Unwrap() http.ResponseWriter }); ok && code == http.StatusSwitchingProtocols {
		raw.Unwrap().WriteHeader(code)
		return
	}
	w.gin.WriteHeader(code)
}

func (w ginUpgradeWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) { return w.gin.Hijack() }

// wsAcceptOptions follows serve --origins: "*" (the default) accepts any
// origin, as the HTTP CORS middleware does; otherwise only the listed ones.
func wsAcceptOptions(origins string) *websocket.AcceptOptions {
	opts := &websocket.AcceptOptions{}
	if strings.TrimSpace(origins) == "" || strings.TrimSpace(origins) == "*" {
		opts.InsecureSkipVerify = true
		return opts
	}
	for _, o := range strings.Split(origins, ",") {
		if o = strings.TrimSpace(o); o != "" {
			opts.OriginPatterns = append(opts.OriginPatterns, strings.TrimPrefix(strings.TrimPrefix(o, "https://"), "http://"))
		}
	}
	return opts
}

// wsSession is one WebSocket connection's state.
type wsSession struct {
	conn   *websocket.Conn
	ctx    context.Context
	cancel context.CancelFunc

	mu        sync.Mutex
	order     []string // response ids, oldest first
	responses map[string][]responsesItem
}

// turn runs one response.create message.
func (s *wsSession) turn(c *gin.Context, data []byte) {
	var head struct {
		Type          string          `json:"type"`
		Stream        json.RawMessage `json:"stream"`
		StreamOptions json.RawMessage `json:"stream_options"`
		Background    json.RawMessage `json:"background"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		s.sendError(http.StatusBadRequest, "the message is not a JSON object: "+err.Error(), "invalid_request", nil)
		return
	}
	if head.Type != "response.create" {
		s.sendError(http.StatusBadRequest, "unknown message type "+strconvQuote(head.Type)+"; send response.create", "invalid_request", "type")
		return
	}
	for name, raw := range map[string]json.RawMessage{"stream": head.Stream, "stream_options": head.StreamOptions, "background": head.Background} {
		if len(raw) > 0 {
			s.sendError(http.StatusBadRequest, name+" must not be sent over WebSocket", "invalid_request", name)
			return
		}
	}
	var req responsesRequest
	if err := json.Unmarshal(data, &req); err != nil {
		s.sendError(http.StatusBadRequest, err.Error(), "invalid_request", nil)
		return
	}

	failed := false
	fail := func(status int, message string, code, param any) {
		failed = true
		s.sendError(status, message, code, param)
	}
	useResponsesErrors(c, fail)
	middleware.WithGIL(func() {
		runResponsesRequest(c, req, responsesTransport{
			ctx:       s.ctx,
			events:    s.sendEvent,
			fail:      fail,
			lookup:    s,
			remember:  s.remember,
			websocket: true,
		})
	})
	// The spec: a continuation that fails evicts the response it continued.
	if failed && req.PreviousResponseID != nil {
		s.evict(*req.PreviousResponseID)
	}
}

func (s *wsSession) sendEvent(eventType string, fields map[string]any) {
	s.send(fields)
}

// sendError writes the spec's WebSocket error envelope. error.code is
// required there, so a missing code falls back to the error type.
func (s *wsSession) sendError(status int, message string, code, param any) {
	if code == nil {
		code = responsesErrorType(status)
	}
	s.send(map[string]any{
		"type":   "error",
		"status": status,
		"error":  responsesErrorPayload(status, message, code, param),
	})
}

func (s *wsSession) send(msg map[string]any) {
	data, err := json.Marshal(msg)
	if err != nil {
		slog.Error("Encoding a WebSocket message", "error", err)
		return
	}
	// Not s.ctx: the connection-limit error is written after it has expired.
	ctx, cancel := context.WithTimeout(context.Background(), wsWriteTimeout)
	defer cancel()
	if err := s.conn.Write(ctx, websocket.MessageText, data); err != nil {
		slog.Debug("WebSocket write failed; closing the connection", "error", err)
		s.cancel() // stops a running turn at its next token
	}
}

// remember keeps a finished response in connection-local state.
func (s *wsSession) remember(id string, conversation []responsesItem) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.responses[id]; !ok {
		s.order = append(s.order, id)
	}
	s.responses[id] = conversation
	for len(s.order) > wsRecentResponses {
		delete(s.responses, s.order[0])
		s.order = s.order[1:]
	}
}

func (s *wsSession) evict(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.responses[id]; !ok {
		return
	}
	delete(s.responses, id)
	for i, other := range s.order {
		if other == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
}

// get looks in the connection first, then in the server-wide store, which
// holds only store:true responses.
func (s *wsSession) get(id string) ([]responsesItem, bool) {
	s.mu.Lock()
	conversation, ok := s.responses[id]
	s.mu.Unlock()
	if ok {
		return append([]responsesItem(nil), conversation...), true
	}
	return responsesStore.get(id)
}

func (s *wsSession) item(id string) (responsesItem, bool) {
	s.mu.Lock()
	for _, conversation := range s.responses {
		for _, item := range conversation {
			if item.ID == id {
				s.mu.Unlock()
				return item, true
			}
		}
	}
	s.mu.Unlock()
	return responsesStore.item(id)
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
