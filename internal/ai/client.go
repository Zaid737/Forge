package ai

import "github.com/openai/openai-go"

type Service struct {
	client openai.Client
}

func NewService(client openai.Client) *Service {
	return &Service{
		client: client,
	}
}
