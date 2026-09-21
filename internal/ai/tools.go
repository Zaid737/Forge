package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

type CalculatorInput struct {
	A         float64 `json:"a"`
	B         float64 `json:"b"`
	Operation string  `json:"operation"`
}

func calculate(input CalculatorInput) float64 {
	switch input.Operation {
	case "add":
		return input.A + input.B

	case "subtract":
		return input.A - input.B

	case "multiply":
		return input.A * input.B

	case "divide":
		return input.A / input.B

	default:
		return 0
	}
}

func getCurrentTime() string {
	return time.Now().Format(
		"Monday, January 2, 2006 3:04:05 PM",
	)
}

func calculatorTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name: "calculator",
			Description: param.NewOpt(
				"Perform basic arithmetic calculations.",
			),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"a": map[string]interface{}{
						"type":        "number",
						"description": "The first number.",
					},
					"b": map[string]interface{}{
						"type":        "number",
						"description": "The second number.",
					},
					"operation": map[string]interface{}{
						"type": "string",
						"enum": []string{
							"add",
							"subtract",
							"multiply",
							"divide",
						},
						"description": "The arithmetic operation.",
					},
				},
				"required": []string{
					"a",
					"b",
					"operation",
				},
				"additionalProperties": false,
			},
		},
	}
}

func currentTimeTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name: "current_time",
			Description: param.NewOpt(
				"Get the current server time.",
			),
			Parameters: openai.FunctionParameters{
				"type":                 "object",
				"properties":           map[string]interface{}{},
				"required":             []string{},
				"additionalProperties": false,
			},
		},
	}
}

func (s *Service) Tools(
	ctx context.Context,
	message string,
) (string, error) {
	tools := []openai.ChatCompletionToolParam{
		calculatorTool(),
		currentTimeTool(),
	}

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(
			"You are ForgeAI. Use the available tools whenever " +
				"they are appropriate for the user's request.",
		),
		openai.UserMessage(message),
	}

	response, err := s.client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Model:    openai.ChatModelGPT4o,
			Messages: messages,
			Tools:    tools,
		},
	)

	if err != nil {
		return "", err
	}

	assistantMessage := response.Choices[0].Message

	if len(assistantMessage.ToolCalls) == 0 {
		return assistantMessage.Content, nil
	}

	messages = append(
		messages,
		assistantMessage.ToParam(),
	)

	for _, toolCall := range assistantMessage.ToolCalls {
		switch toolCall.Function.Name {
		case "calculator":
			var input CalculatorInput

			if err := json.Unmarshal(
				[]byte(toolCall.Function.Arguments),
				&input,
			); err != nil {
				return "", fmt.Errorf(
					"invalid calculator arguments: %w",
					err,
				)
			}

			result := calculate(input)

			messages = append(
				messages,
				openai.ToolMessage(
					strconv.FormatFloat(
						result,
						'f',
						-1,
						64,
					),
					toolCall.ID,
				),
			)

		case "current_time":
			result := getCurrentTime()

			messages = append(
				messages,
				openai.ToolMessage(
					result,
					toolCall.ID,
				),
			)
		}
	}

	finalResponse, err := s.client.Chat.Completions.New(
		ctx,
		openai.ChatCompletionNewParams{
			Model:    openai.ChatModelGPT4o,
			Messages: messages,
			Tools:    tools,
		},
	)

	if err != nil {
		return "", err
	}

	return finalResponse.Choices[0].Message.Content, nil
}
