package ai

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

func (s *Service) CreateEmbedding(
	ctx context.Context,
	text string,
) ([]float64, error) {
	response, err := s.client.Embeddings.New(
		ctx,
		openai.EmbeddingNewParams{
			Model: openai.EmbeddingModelTextEmbedding3Small,
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: param.NewOpt(text),
			},
		},
	)

	if err != nil {
		return nil, err
	}

	return response.Data[0].Embedding, nil
}
