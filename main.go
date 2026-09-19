package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
)

type ExtractedPerson struct {
	Name     *string `json:"name"`
	Age      *int    `json:"age"`
	JobTitle *string `json:"job_title"`
	Company  *string `json:"company"`
}

type ChatRequest struct {
	ConversationId string `json:"conversation_id" binding:"required"`
	Message        string `json:"message" binding:"required"`
}

type ChatResponse struct {
	Answer string `json:"answer"`
}

type Conversation struct {
	Messages []openai.ChatCompletionMessageParamUnion
}

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env found")
	}

	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		log.Fatal("API key not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	conversations := make(map[string]*Conversation)

	router := gin.Default()

	router.POST("/v1/ai/chat", func(c *gin.Context) {
		var req ChatRequest

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": err.Error(),
			})
			return
		}

		conversation, exists := conversations[req.ConversationId]

		if !exists {
			conversation = &Conversation{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage(
						"You are ForgeAI, a helpful technical assistant. " +
							"Explain technical concepts clearly and concisely. " +
							"Use examples when they improve understanding.",
					),
				},
			}

			conversations[req.ConversationId] = conversation
		}

		conversation.Messages = append(
			conversation.Messages,
			openai.UserMessage(req.Message),
		)

		response, err := client.Chat.Completions.New(
			context.Background(),
			openai.ChatCompletionNewParams{
				Model:    openai.ChatModelGPT4o,
				Messages: conversation.Messages,
			},
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "failed to generate response",
			})
			return
		}

		answer := response.Choices[0].Message.Content

		conversation.Messages = append(
			conversation.Messages,
			openai.AssistantMessage(answer),
		)

		c.JSON(http.StatusOK, ChatResponse{
			Answer: answer,
		})
	})

	router.POST("/v1/ai/extract", func(c *gin.Context) {
		var req struct {
			Text string `json:"text" binding:"required"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": err.Error(),
			})
			return
		}

		personSchema := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "string",
				},
				"age": map[string]interface{}{
					"type": "integer",
				},
				"job_title": map[string]interface{}{
					"type": "string",
				},
				"company": map[string]interface{}{
					"type": "string",
				},
			},
			"required": []string{
				"name",
				"age",
				"job_title",
				"company",
			},
			"additionalProperties": false,
		}

		response, err := client.Chat.Completions.New(
			context.Background(),
			openai.ChatCompletionNewParams{
				Model: openai.ChatModelGPT4o,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage(
						"Extract the person's information from the provided text. " +
							"Return the result according to the provided JSON schema.",
					),
					openai.UserMessage(req.Text),
				},
				ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
					OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
						JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
							Name:        "person_extraction",
							Description: param.NewOpt("Extracted information about a person"),
							Schema:      personSchema,
							Strict:      param.NewOpt(true),
						},
					},
				},
			},
		)

		if err != nil {
			log.Println("OpenAI error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		var person ExtractedPerson

		err = json.Unmarshal(
			[]byte(response.Choices[0].Message.Content),
			&person,
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "failed to parse structured response",
			})
			return
		}

		c.JSON(http.StatusOK, person)
	})

	log.Println("Server running on http://localhost:8080")

	if err := router.Run(":8080"); err != nil {
		log.Fatal(err)
	}
}
