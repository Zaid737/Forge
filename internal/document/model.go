package document

type Document struct {
	ID         int64
	Content    string
	Similarity float64
}

type CreateRequest struct {
	Content   string `json:"content" binding:"required"`
	ChunkSize int    `json:"chunk_size"`
	Overlap   int    `json:"overlap"`
}

type CreateResponse struct {
	IDs    []int64 `json:"ids"`
	Chunks int     `json:"chunks"`
}

type SearchRequest struct {
	Query string `json:"query" binding:"required"`
}

type SearchResult struct {
	ID         int64   `json:"id"`
	Content    string  `json:"content"`
	Similarity float64 `json:"similarity"`
}

type RAGRequest struct {
	Query     string  `json:"query" binding:"required"`
	TopK      int     `json:"top_k"`
	Threshold float64 `json:"threshold"`
}

type RAGResponse struct {
	Answer string `json:"answer"`
}
