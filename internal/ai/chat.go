package ai

import (
	"context"

	"github.com/openai/openai-go"
)

func (s *Service) Chat(
	ctx context.Context,
	message string,
) (string, error) {
	response, err := s.client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(
					"You are ForgeAI, a helpful AI assistant.",
				),
				openai.UserMessage(message),
			},
		},
	)

	if err != nil {
		return "", err
	}

	return response.Choices[0].Message.Content, nil
}

func (s *Service) Extract(
	ctx context.Context,
	text string,
) (string, error) {
	response, err := s.client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(
					"Extract structured information from the user's text.",
				),
				openai.UserMessage(text),
			},
		},
	)

	if err != nil {
		return "", err
	}

	return response.Choices[0].Message.Content, nil
}

func (s *Service) AnswerWithContext(
	ctx context.Context,
	query string,
	contextText string,
) (string, error) {
	response, err := s.client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Model: openai.ChatModelGPT4o,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(
					"You are ForgeAI, a helpful RAG assistant. " +
						"Answer questions using only the supplied context. " +
						"Do not invent information.",
				),
				openai.UserMessage(
					"Context:\n\n" +
						contextText +
						"\n\nQuestion:\n" +
						query +
						"\n\nIf the answer is not present in the context, " +
						"say that the information is not available.",
				),
			},
		},
	)

	if err != nil {
		return "", err
	}

	return response.Choices[0].Message.Content, nil
}
