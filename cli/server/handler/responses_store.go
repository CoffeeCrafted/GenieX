// Copyright (c) 2026 CoffeeCrafted
// SPDX-License-Identifier: BSD-3-Clause

package handler

import "sync"

// responseStore keeps stored responses in memory for previous_response_id and
// item_reference: each response's whole conversation (input and output, in
// input-item form) under its id, and every item under its own id. It holds the
// most recent maxResponses responses and forgets the oldest; a server restart
// forgets them all, which a client sees as previous_response_not_found.
type responseStore struct {
	mu        sync.Mutex
	max       int
	order     []string // response ids, oldest first
	responses map[string][]responsesItem
	items     map[string]responsesItem
	itemsOf   map[string][]string // response id -> ids of the items it added
}

// maxStoredResponses bounds memory: a conversation of a few hundred turns is
// kept whole by its latest response, so this many covers many clients.
const maxStoredResponses = 256

var responsesStore = newResponseStore(maxStoredResponses)

func newResponseStore(max int) *responseStore {
	return &responseStore{
		max:       max,
		responses: map[string][]responsesItem{},
		items:     map[string]responsesItem{},
		itemsOf:   map[string][]string{},
	}
}

// put stores conversation (the input items, then the output items) as id.
func (s *responseStore) put(id string, conversation []responsesItem) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.responses[id]; !ok {
		s.order = append(s.order, id)
	}
	s.responses[id] = conversation
	for _, item := range conversation {
		if item.ID != "" {
			s.items[item.ID] = item
			s.itemsOf[id] = append(s.itemsOf[id], item.ID)
		}
	}
	for len(s.order) > s.max {
		s.drop(s.order[0])
	}
}

// delete forgets response id and the items no other response holds.
func (s *responseStore) delete(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.drop(id)
}

// drop is delete with s.mu held.
func (s *responseStore) drop(id string) {
	if _, ok := s.responses[id]; !ok {
		return
	}
	delete(s.responses, id)
	for _, itemID := range s.itemsOf[id] {
		if !s.referencedElsewhere(itemID, id) {
			delete(s.items, itemID)
		}
	}
	delete(s.itemsOf, id)
	for i, other := range s.order {
		if other == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
}

// referencedElsewhere reports whether a response other than except still
// holds itemID; a later turn of the same conversation repeats earlier items.
func (s *responseStore) referencedElsewhere(itemID, except string) bool {
	for id, ids := range s.itemsOf {
		if id == except {
			continue
		}
		for _, other := range ids {
			if other == itemID {
				return true
			}
		}
	}
	return false
}

func (s *responseStore) get(id string) ([]responsesItem, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	conversation, ok := s.responses[id]
	return append([]responsesItem(nil), conversation...), ok
}

func (s *responseStore) item(id string) (responsesItem, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	item, ok := s.items[id]
	return item, ok
}
