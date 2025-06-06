// Copyright 2025 Google LLC
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
//
// SPDX-License-Identifier: Apache-2.0

package base

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"reflect"
	"regexp"
	"strings"

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
	r := jsonschema.Reflector{
		DoNotReference: true,
		Mapper: func(t reflect.Type) *jsonschema.Schema {
			// []any generates `{ type: "array", items: true }` which is not valid JSON schema.
			if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Interface {
				return &jsonschema.Schema{
					Type: "array",
					Items: &jsonschema.Schema{
						// This field is not necessary but it's the most benign way for the object to not be empty.
						AdditionalProperties: jsonschema.TrueSchema,
					},
				}
			}
			return nil // Return nil to use default schema generation for other types
		},
	}
	s = r.Reflect(x)
	// TODO: Unwind this change once Monaco Editor supports newer than JSON schema draft-07.
	s.Version = ""
	return s
}

// SchemaAsMap converts json schema struct to a map (JSON representation).
func SchemaAsMap(s *jsonschema.Schema) map[string]any {
	jsb, err := s.MarshalJSON()
	if err != nil {
		log.Panicf("failed to marshal schema: %v", err)
	}

	// Check if the marshaled JSON is "true" (indicates an empty schema)
	if string(jsb) == "true" {
		return make(map[string]any)
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

// GetJsonLines splits a string by newlines, trims whitespace from each line,
// and returns a slice containing only the lines that start with the prefix.
func GetJsonLines(text string, prefix string) []string {
	jsonText := ExtractJSONFromMarkdown(text)

	// Handle both actual "\n" newline strings, as well as newline bytes
	jsonText = strings.ReplaceAll(jsonText, "\n", `\n`)

	// Split the input string into lines based on the newline character.
	lines := strings.Split(jsonText, `\n`)

	var result []string
	for _, line := range lines {
		if line == "" {
			continue
		}

		// Trim leading and trailing whitespace from the current line.
		trimmedLine := strings.TrimSpace(line)
		if strings.HasPrefix(trimmedLine, prefix) {
			result = append(result, trimmedLine)
		}
	}

	// Return the slice containing the filtered and trimmed lines.
	return result
}

// Looks for a JsonObject in the given string, returns empty string if none found
func GetJsonObject(text string) (result string) {
	// Trim leading and trailing whitespace from the current line.
	trimmedLine := strings.TrimSpace(text)
	// Check if the trimmed line starts with the character '{'.
	if strings.HasPrefix(trimmedLine, "{") {
		result = trimmedLine
	}

	return result
}
