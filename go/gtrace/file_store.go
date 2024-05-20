// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gtrace

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"

	"github.com/firebase/genkit/go/internal"
)

// A FileStore is a Store that writes traces to files.
type FileStore struct {
	dir string
}

// NewFileStore creates a FileStore that writes traces to the given
// directory. The directory is created if it does not exist.
func NewFileStore(dir string) (*FileStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &FileStore{dir: dir}, nil
}

// Save implements [Store.Save].
// It is not safe to call Save concurrently with the same ID.
func (s *FileStore) Save(ctx context.Context, id string, td *Data) error {
	existing, err := s.Load(ctx, id)
	if err == nil {
		// Merge the existing spans with the incoming ones.
		// Mutate existing because we know it has no other references.
		for k, v := range td.Spans {
			existing.Spans[k] = v
		}
		existing.TraceID = id
		existing.DisplayName = td.DisplayName
		existing.StartTime = td.StartTime
		existing.EndTime = td.EndTime
		td = existing
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return internal.WriteJSONFile(filepath.Join(s.dir, internal.Clean(id)), td)
}

// Load implements [Store.Load].
func (s *FileStore) Load(ctx context.Context, id string) (*Data, error) {
	var td *Data
	if err := internal.ReadJSONFile(filepath.Join(s.dir, internal.Clean(id)), &td); err != nil {
		return nil, err
	}
	return td, nil
}

// List implements [Store.List].
// The traces are returned in the order they were written, newest first.
// The default limit is 10.
func (s *FileStore) List(ctx context.Context, q *Query) ([]*Data, string, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, "", err
	}
	// Sort by modified time.
	modTimes := map[string]time.Time{}
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			return nil, "", err
		}
		modTimes[e.Name()] = info.ModTime()
	}
	slices.SortFunc(entries, func(e1, e2 os.DirEntry) int {
		return modTimes[e2.Name()].Compare(modTimes[e1.Name()])
	})

	// Determine subsequence to return.
	start, end, err := listRange(q, len(entries))
	if err != nil {
		return nil, "", err
	}

	var ts []*Data
	for _, e := range entries[start:end] {
		var t *Data
		if err := internal.ReadJSONFile(filepath.Join(s.dir, e.Name()), &t); err != nil {
			return nil, "", err
		}
		ts = append(ts, t)
	}
	ctoken := ""
	if end < len(entries) {
		ctoken = strconv.Itoa(end)
	}
	return ts, ctoken, nil
}

// listRange returns the range of elements to return from a List call.
func listRange(q *Query, total int) (start, end int, err error) {
	const defaultLimit = 10
	start = 0
	end = total
	limit := 0
	ctoken := ""
	if q != nil {
		limit = q.Limit
		ctoken = q.ContinuationToken
	}
	if ctoken != "" {
		// A continuation token is just an integer index in string form.
		// This doesn't work well with newest-first order if files are added during listing,
		// because the indexes will change.
		// But we use it for consistency with the javascript implementation.
		// TODO(jba): consider using distance from the end (len(entries) - end).
		start, err = strconv.Atoi(ctoken)
		if err != nil {
			return 0, 0, fmt.Errorf("%w: parsing continuation token: %v", ErrBadQuery, err)
		}
		if start < 0 || start >= total {
			return 0, 0, fmt.Errorf("%w: continuation token out of range", ErrBadQuery)
		}
	}
	if limit < 0 {
		return 0, 0, fmt.Errorf("%w: negative limit", ErrBadQuery)
	}
	if limit == 0 {
		limit = defaultLimit
	}
	end = start + limit
	if end > total {
		end = total
	}
	return start, end, nil
}

func (s *FileStore) LoadAny(id string, p any) error {
	return internal.ReadJSONFile(filepath.Join(s.dir, internal.Clean(id)), p)
}
