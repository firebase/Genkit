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

package firebase

import (
	"context"
	"sync"

	firebase "firebase.google.com/go/v4"
	"firebase.google.com/go/v4/auth"
)

type FirebaseApp interface {
	Auth(ctx context.Context) (*auth.Client, error)
}

var (
	app   *firebase.App
	mutex sync.Mutex
)

// app returns a cached Firebase app.
func App(ctx context.Context) (FirebaseApp, error) {
	mutex.Lock()
	defer mutex.Unlock()
	if app == nil {
		newApp, err := firebase.NewApp(ctx, nil)
		if err != nil {
			return nil, err
		}
		app = newApp
	}
	return app, nil
}
