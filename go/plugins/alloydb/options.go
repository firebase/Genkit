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
	"github.com/jackc/pgx/v5/pgxpool"
)

// Option is a function type that can be used to modify the Engine.
type Option func(p *engineConfig)

type engineConfig struct {
	projectID       string
	region          string
	cluster         string
	instance        string
	connPool        *pgxpool.Pool
	database        string
	user            string
	password        string
	ipType          IpType
	iamAccountEmail string
	userAgents      string
}

// WithAlloyDBInstance sets the project, region, and instance fields.
func WithAlloyDBInstance(projectID, region, cluster, instance string) Option {
	return func(p *engineConfig) {
		p.projectID = projectID
		p.region = region
		p.instance = instance
		p.cluster = cluster
	}
}

// WithPool sets the Port field.
func WithPool(pool *pgxpool.Pool) Option {
	return func(p *engineConfig) {
		p.connPool = pool
	}
}

// WithDatabase sets the Database field.
func WithDatabase(database string) Option {
	return func(p *engineConfig) {
		p.database = database
	}
}

// WithUser sets the User field.
func WithUser(user string) Option {
	return func(p *engineConfig) {
		p.user = user
	}
}

// WithPassword sets the Password field.
func WithPassword(password string) Option {
	return func(p *engineConfig) {
		p.password = password
	}
}

// WithIPType sets the IpType field.
func WithIPType(ipType IpType) Option {
	return func(p *engineConfig) {
		p.ipType = ipType
	}
}

// WithIAMAccountEmail sets the IAMAccountEmail field.
func WithIAMAccountEmail(email string) Option {
	return func(p *engineConfig) {
		p.iamAccountEmail = email
	}
}
