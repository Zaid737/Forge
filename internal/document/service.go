package document

import (
	"context"
	"strings"

	"forge-ai/internal/ai"
)

type AI interface {
	CreateEmbedding(
		ctx context.Context,
		text string,
	) ([]float64, error)

	AnswerWithContext(
		ctx context.Context,
		query string,
		contextText string,
	) (string, error)
}

type Service struct {
	repository *Repository
	ai         AI
}

func NewService(
	repository *Repository,
	aiService AI,
) *Service {
	return &Service{
		repository: repository,
		ai:         aiService,
	}
}

func (s *Service) Create(
	ctx context.Context,
	req CreateRequest,
) (CreateResponse, error) {
	if req.ChunkSize <= 0 {
		req.ChunkSize = 1000
	}

	if req.Overlap < 0 || req.Overlap >= req.ChunkSize {
		req.Overlap = 200
	}

	chunks := ai.ChunkText(
		req.Content,
		req.ChunkSize,
		req.Overlap,
	)

	if len(chunks) == 0 {
		return CreateResponse{}, nil
	}

	documents := make(
		[]Document,
		0,
		len(chunks),
	)

	embeddings := make(
		[][]float64,
		0,
		len(chunks),
	)

	for _, chunk := range chunks {
		embedding, err := s.ai.CreateEmbedding(
			ctx,
			chunk,
		)

		if err != nil {
			return CreateResponse{}, err
		}

		documents = append(
			documents,
			Document{
				Content: chunk,
			},
		)

		embeddings = append(
			embeddings,
			embedding,
		)
	}

	ids, err := s.repository.CreateMany(
		ctx,
		documents,
		embeddings,
	)

	if err != nil {
		return CreateResponse{}, err
	}

	return CreateResponse{
		IDs:    ids,
		Chunks: len(chunks),
	}, nil
}

func (s *Service) Search(
	ctx context.Context,
	req SearchRequest,
) ([]SearchResult, error) {
	embedding, err := s.ai.CreateEmbedding(
		ctx,
		req.Query,
	)

	if err != nil {
		return nil, err
	}

	return s.repository.Search(
		ctx,
		embedding,
		5,
	)
}

func (s *Service) RAG(
	ctx context.Context,
	req RAGRequest,
) (RAGResponse, error) {
	if req.TopK <= 0 {
		req.TopK = 5
	}

	if req.TopK > 20 {
		req.TopK = 20
	}

	if req.Threshold <= 0 {
		req.Threshold = 0.3
	}

	embedding, err := s.ai.CreateEmbedding(
		ctx,
		req.Query,
	)

	if err != nil {
		return RAGResponse{}, err
	}

	results, err := s.repository.SearchForRAG(
		ctx,
		embedding,
		req.TopK,
		req.Threshold,
	)

	if err != nil {
		return RAGResponse{}, err
	}

	contextParts := make(
		[]string,
		0,
		len(results),
	)

	for _, result := range results {
		contextParts = append(
			contextParts,
			result.Content,
		)
	}

	contextText := strings.Join(
		contextParts,
		"\n\n",
	)

	answer, err := s.ai.AnswerWithContext(
		ctx,
		req.Query,
		contextText,
	)

	if err != nil {
		return RAGResponse{}, err
	}

	return RAGResponse{
		Answer: answer,
	}, nil
}
