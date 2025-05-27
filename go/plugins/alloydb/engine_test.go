// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package alloydb

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/assert"
)

func TestApplyEngineOptionsConfig(t *testing.T) {

	testCases := []struct {
		name       string
		opts       []Option
		wantErr    bool
		wantIpType IpType
	}{
		{
			name: "valid config with connection pool",
			opts: []Option{
				WithPool(&pgxpool.Pool{}),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "valid config with instance details",
			opts: []Option{
				WithAlloyDBInstance("testproject", "testregion", "testcluster", "testinstance"),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "missing database",
			opts: []Option{
				WithAlloyDBInstance("testproject", "testregion", "testcluster", "testinstance"),
			},
			wantErr:    true,
			wantIpType: PUBLIC,
		},
		{
			name: "missing all connection details",
			opts: []Option{
				WithDatabase("testdb"),
			},
			wantErr:    true,
			wantIpType: PUBLIC,
		},
		{
			name: "ip type private",
			opts: []Option{
				WithAlloyDBInstance("testproject", "testregion", "testcluster", "testinstance"),
				WithDatabase("testdb"),
				WithIPType(PRIVATE),
			},
			wantErr:    false,
			wantIpType: PRIVATE,
		},
		{
			name: "custom EmailRetriever",
			opts: []Option{
				WithAlloyDBInstance("testproject", "testregion", "testcluster", "testinstance"),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := applyEngineOptions(tc.opts)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.wantIpType, cfg.ipType)
			}
		})
	}
}

func TestGetUser(t *testing.T) {
	testCases := []struct {
		name        string
		cfg         engineConfig
		wantUser    string
		wantIAMAuth bool
		wantErr     bool
	}{
		{
			name: "user and password provided",
			cfg: engineConfig{
				user:     "testuser",
				password: "testpassword",
			},
			wantUser:    "testuser",
			wantIAMAuth: false,
			wantErr:     false,
		},
		{
			name: "iam account email provided",
			cfg: engineConfig{
				iamAccountEmail: "iam@example.com",
			},
			wantUser:    "iam@example.com",
			wantIAMAuth: true,
			wantErr:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			user, iamAuth, err := getUser(ctx, tc.cfg)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.wantUser, user)
				assert.Equal(t, tc.wantIAMAuth, iamAuth)
			}
		})
	}
}
