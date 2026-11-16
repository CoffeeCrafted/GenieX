// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

// The chat handlers' error bodies are {"error": "<message>"} or
// {"error": "<message>", "code": <SDK code>}. The Responses endpoint shares
// their validation and model loading but must answer in the Open Responses
// error shape, over HTTP as a JSON body and over WebSocket as an error
// envelope, so every shared error goes through writeAPIError, and a Responses
// request stores its own writer in the context under responsesErrorWriter.
const responsesErrorWriter = "geniex.responses_error_writer"

// errorWriter writes one Open Responses error: status, message, and code and
// param as strings or nil.
type errorWriter func(status int, message string, code, param any)

// useResponsesErrors makes the shared error paths of this request write w.
func useResponsesErrors(c *gin.Context, w errorWriter) {
	c.Set(responsesErrorWriter, w)
}

// writeAPIError writes body unchanged (the chat endpoints' bytes stay the same)
// unless the request is a Responses request, which gets the spec's shape.
func writeAPIError(c *gin.Context, status int, body map[string]any) {
	w, ok := c.Value(responsesErrorWriter).(errorWriter)
	if !ok {
		c.JSON(status, body)
		return
	}
	message, _ := body["error"].(string)
	var code any
	if raw, ok := body["code"]; ok && raw != nil {
		code = fmt.Sprint(raw)
	}
	w(status, message, code, nil)
}

// httpResponsesErrors writes Open Responses errors as an HTTP JSON body.
func httpResponsesErrors(c *gin.Context) errorWriter {
	return func(status int, message string, code, param any) {
		writeResponsesError(c, status, message, code, param)
	}
}

// writeResponsesError writes an Open Responses error body. code and param are
// strings or nil; type follows the status as the spec's error table lists it.
func writeResponsesError(c *gin.Context, status int, message string, code, param any) {
	c.JSON(status, map[string]any{"error": responsesErrorPayload(status, message, code, param)})
}

func responsesErrorPayload(status int, message string, code, param any) map[string]any {
	return map[string]any{
		"type":    responsesErrorType(status),
		"message": message,
		"code":    code,
		"param":   param,
	}
}

func responsesErrorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return "invalid_request"
	case http.StatusNotFound:
		return "not_found"
	case http.StatusTooManyRequests:
		return "too_many_requests"
	default:
		return "server_error"
	}
}
