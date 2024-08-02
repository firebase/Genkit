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

package base

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"

	"github.com/invopop/jsonschema"
)

// JSONString returns json.Marshal(x) as a string. If json.Marshal returns
// an error, jsonString returns the error text as a JSON string beginning "ERROR:".
func JSONString(x any) string {
	bytes, err := json.Marshal(x)
	if err != nil {
		bytes, _ = json.Marshal(fmt.Sprintf("ERROR: %v", err))
	}
	return string(bytes)
}

// PrettyJSONString returns json.MarshalIndent(x, "", "  ") as a string.
// If json.MarshalIndent returns an error, jsonString returns the error text as
// a JSON string beginning "ERROR:".
func PrettyJSONString(x any) string {
	bytes, err := json.MarshalIndent(x, "", "  ")
	if err != nil {
		bytes, _ = json.MarshalIndent(fmt.Sprintf("ERROR: %v", err), "", "  ")
	}
	return string(bytes)
}

// WriteJSONFile writes value to filename as JSON.
func WriteJSONFile(filename string, value any) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, f.Close())
	}()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "    ") // make the value easy to read for debugging
	return enc.Encode(value)
}

// ReadJSONFile JSON-decodes the contents of filename into pvalue,
// which must be a pointer.
func ReadJSONFile(filename string, pvalue any) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewDecoder(f).Decode(pvalue)
}

func InferJSONSchema(x any) (s *jsonschema.Schema) {
	r := jsonschema.Reflector{}
	s = r.Reflect(x)
	// TODO: Unwind this change once Monaco Editor supports newer than JSON schema draft-07.
	s.Version = ""
	return s
}

func InferJSONSchemaNonReferencing(x any) (s *jsonschema.Schema) {
	r := jsonschema.Reflector{
		DoNotReference: true,
	}
	s = r.Reflect(x)
	// TODO: Unwind this change once Monaco Editor supports newer than JSON schema draft-07.
	s.Version = ""
	return s
}

// SchemaAsMap convers json schema struct to a map (JSON representation).
func SchemaAsMap(s *jsonschema.Schema) map[string]any {
	jsb, err := s.MarshalJSON()
	if err != nil {
		log.Panicf("failed to marshal schema: %v", err)
	}
	var m map[string]any
	err = json.Unmarshal(jsb, &m)
	if err != nil {
		log.Panicf("failed to unmarshal schema: %v", err)
	}
	return m
}

var jsonMarkdownRegex = regexp.MustCompile("```(json)?((\n|.)*?)```")

// ExtractJSONFromMarkdown returns the contents of the first fenced code block in
// the markdown text md. If there is none, it returns md.
func ExtractJSONFromMarkdown(md string) string {
	// TODO: improve this
	matches := jsonMarkdownRegex.FindStringSubmatch(md)
	if matches == nil {
		return md
	}
	return matches[2]
}
