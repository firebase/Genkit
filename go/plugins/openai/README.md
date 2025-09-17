# OpenAI Plugin

This plugin provides a plugin for the OpenAI API.

## Test the plugin

```bash
UNIMPLEMENTED
```

## Test the client

Unit tests:

```bash
go test -v ./client
```

Live tests (with `OPENAI_API_KEY` env variable):

```bash
export OPENAI_API_KEY="your-api-key"
go test -v ./client -test-live
```

Live tests (setting openai api key in flag):

```bash
go test -v ./client -test-live -api-key="your-api-key"
```