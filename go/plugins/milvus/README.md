# Milvus plugin

## Configuration

To use this plugin, first create a `MilvusEngine` instance:

```go
// Create MilvusEngine instance with basic authentication
engine, err := NewMilvusEngine(ctx, WithAddress("localhost:19530"), WithUsername("username"), WithPassword("password"))
if err != nil {
    return err
}
```

```go
// Create MilvusEngine instance with API Key authentication
engine, err := NewMilvusEngine(ctx, WithAddress("localhost:19530"), WithAPIKey("APIKey"))
if err != nil {
    return err
}
```

Then, specify the plugin when you initialize Genkit:

```go
milvus := &Milvus{
    Engine: engine,
}
g, err := genkit.Init(ctx, genkit.WithPlugins(milvus))
if err != nil {
    return err
}
```

## Configuration and Usage

To configure and use the retriever, define a collection configuration:

```go
// Define collection configuration
cfg := &CollectionConfig{
  Name: "documents",
  IdKey: "id",
  ScoreKey: "score",
  VectorKey: "embedding",
  TextKey: "content",
  VectorDim: 768,
  Embedder: embedder,
  EmbedderOptions: nil,
}

// Define retriever with the configuration
docStore, retrieval, err := DefineRetriever(ctx, g, cfg, &ai.RetrieverOptions{})
if err != nil {
    return err
}
```

### Indexing Documents

```go
// Prepare documents to index
docs := []*ai.Document{
ai.DocumentFromText("text1", map[string]any{"id": int64(1)}),
ai.DocumentFromText("text2", map[string]any{"id": int64(2)}),
}

if err := docStore.Index(ctx, docs); err != nil {
    return err
}
```

### Retrieving Documents

```go
// Create query document
queryDoc := ai.DocumentFromText("AI capabilities", nil)

// Retrieve similar documents
retrieveDocs, err := genkit.Retrieve(ctx, g,
  ai.WithRetriever(retrieval),
  ai.WithDocs(queryDoc),
  ai.WithConfig(&milvus.RetrieverOptions{
    Limit: 2,
  }),
)
```

## Engine Options

The following options are available when creating a `MilvusEngine`:

- **WithAddress(address string)**: Sets the Milvus server address (required)
- **WithUsername(username string)**: Sets username for authentication
- **WithPassword(password string)**: Sets password for authentication
- **WithDbName(dbName string)**: Selects a specific database
- **WithEnableTlsAuth(enable bool)**: Enables TLS authentication
- **WithAPIKey(apiKey string)**: Sets API key for managed services
- **WithDialOptions(opts ...grpc.DialOption)**: Adds custom gRPC options
- **WithDisableConn(disable bool)**: Prevents connection establishment (for
  testing)
- **WithServerVersion(version string)**: Specifies expected server version

## Collection Configuration

The `CollectionConfig` struct requires the following fields:

- **Name**: Milvus collection name
- **IdKey**: Column name storing document identifiers
- **ScoreKey**: Metadata key for similarity scores in results
- **VectorKey**: Column name storing embedding vectors
- **TextKey**: Column name storing original document text
- **VectorDim**: Dimensionality of embedding vectors
- **Embedder**: Embedding model for queries and documents
- **EmbedderOptions**: Options passed to the embedder (optional)

### Retriever Options

The `RetrieverOptions` struct requires the following fields:

- **Limit**: Limit is the maximum number of results to retrieve (https://milvus.io/docs/single-vector-search.md#Use-Limit-and-Offset).
- **Columns***: Columns list additional scalar fields to return with each result (https://milvus.io/docs/single-vector-search.md#Use-Output-Fields).
- **Partitions**: Partitions restrict the search to the specified partition names (https://milvus.io/docs/single-vector-search.md#ANN-Search-in-Partition).
- **Offset**: Offset skips the first N results (https://milvus.io/docs/single-vector-search.md#Use-Limit-and-Offset).
- **Filter**: Filter is a boolean expression to post-filter rows (https://milvus.io/docs/filtered-search.md#Search-with-standard-filtering).
- **FilterOptions**: FilterOptions passes engine-specific search parameters (https://milvus.io/docs/filtered-search.md#Search-with-iterative-filtering).
