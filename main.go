package main

import (
	"context"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/joho/godotenv"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
)

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

type EmbedRequest struct {
	Text string `json:"text" binding:"required"`
}

type EmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

type DocumentRequest struct {
	Content   string `json:"content" binding:"required"`
	ChunkSize int    `json:"chunk_size"`
	Overlap   int    `json:"overlap"`
}

type DocumentResponse struct {
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

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct float64
	var magnitudeA float64
	var magnitudeB float64

	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}

	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB))
}

func createEmbedding(
	ctx context.Context,
	client openai.Client,
	text string,
) ([]float64, error) {
	response, err := client.Embeddings.New(
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

func chunkText(text string, chunkSize, overlap int) []string {
	if chunkSize <= 0 {
		return nil
	}

	if overlap < 0 || overlap >= chunkSize {
		overlap = 0
	}

	var chunks []string

	start := 0

	for start < len(text) {
		end := start + chunkSize

		if end > len(text) {
			end = len(text)
		}

		chunks = append(chunks, text[start:end])

		if end == len(text) {
			break
		}

		start = end - overlap
	}

	return chunks
}

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found")
	}

	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	dbURL := "postgres://" +
		os.Getenv("DB_USER") + ":" +
		os.Getenv("DB_PASSWORD") + "@" +
		os.Getenv("DB_HOST") + ":" +
		os.Getenv("DB_PORT") + "/" +
		os.Getenv("DB_NAME") +
		"?sslmode=" + os.Getenv("DB_SSLMODE")

	dbConfig, err := pgx.ParseConfig(dbURL)
	if err != nil {
		log.Fatal("Invalid PostgreSQL configuration:", err)
	}

	db, err := pgx.ConnectConfig(context.Background(), dbConfig)
	if err != nil {
		log.Fatal("PostgreSQL connection failed:", err)
	}

	defer db.Close(context.Background())

	if err := db.Ping(context.Background()); err != nil {
		log.Fatal("PostgreSQL ping failed:", err)
	}

	log.Println("Connected to PostgreSQL")

	router := gin.Default()

	conversations := make(map[string]*Conversation)

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
			log.Println("OpenAI error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
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
					"type": []string{"string", "null"},
				},
				"age": map[string]interface{}{
					"type": []string{"integer", "null"},
				},
				"job_title": map[string]interface{}{
					"type": []string{"string", "null"},
				},
				"company": map[string]interface{}{
					"type": []string{"string", "null"},
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
						"Extract information about a person from the provided text. " +
							"Return the information as JSON. " +
							"Use null when information is not available.",
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

		c.JSON(http.StatusOK, gin.H{
			"data": response.Choices[0].Message.Content,
		})
	})

	router.POST("/v1/ai/embed", func(c *gin.Context) {
		var req EmbedRequest

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": err.Error(),
			})
			return
		}

		embedding, err := createEmbedding(
			c.Request.Context(),
			client,
			req.Text,
		)

		if err != nil {
			log.Println("OpenAI error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, EmbedResponse{
			Embedding: embedding,
		})
	})

	router.POST("/v1/ai/documents", func(c *gin.Context) {
		var req DocumentRequest

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": err.Error(),
			})
			return
		}

		chunkSize := req.ChunkSize

		if chunkSize == 0 {
			chunkSize = 1000
		}

		overlap := req.Overlap

		if overlap == 0 {
			overlap = 200
		}

		if overlap >= chunkSize {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "overlap must be smaller than chunk size",
			})
			return
		}

		chunks := chunkText(
			req.Content,
			chunkSize,
			overlap,
		)

		if len(chunks) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "content could not be chunked",
			})
			return
		}

		ids := make([]int64, 0, len(chunks))

		tx, err := db.Begin(c.Request.Context())
		if err != nil {
			log.Println("Transaction error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		defer tx.Rollback(c.Request.Context())

		for _, chunk := range chunks {
			embedding, err := createEmbedding(
				c.Request.Context(),
				client,
				chunk,
			)

			if err != nil {
				log.Println("Embedding error:", err)

				c.JSON(http.StatusInternalServerError, gin.H{
					"error": err.Error(),
				})
				return
			}

			vector := "["

			for i, value := range embedding {
				if i > 0 {
					vector += ","
				}

				vector += strconv.FormatFloat(
					value,
					'f',
					-1,
					64,
				)
			}

			vector += "]"

			var id int64

			err = tx.QueryRow(
				c.Request.Context(),
				`INSERT INTO documents (content, embedding)
				 VALUES ($1, $2::vector)
				 RETURNING id`,
				chunk,
				vector,
			).Scan(&id)

			if err != nil {
				log.Println("Database error:", err)

				c.JSON(http.StatusInternalServerError, gin.H{
					"error": err.Error(),
				})
				return
			}

			ids = append(ids, id)
		}

		if err := tx.Commit(c.Request.Context()); err != nil {
			log.Println("Transaction commit error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusCreated, DocumentResponse{
			IDs:    ids,
			Chunks: len(chunks),
		})
	})

	router.POST("/v1/ai/search", func(c *gin.Context) {
		var req SearchRequest

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": err.Error(),
			})
			return
		}

		embedding, err := createEmbedding(
			c.Request.Context(),
			client,
			req.Query,
		)

		if err != nil {
			log.Println("Embedding error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		vector := "["

		for i, value := range embedding {
			if i > 0 {
				vector += ","
			}

			vector += strconv.FormatFloat(
				value,
				'f',
				-1,
				64,
			)
		}

		vector += "]"

		rows, err := db.Query(
			c.Request.Context(),
			`SELECT
				id,
				content,
				1 - (embedding <=> $1::vector) AS similarity
			FROM documents
			ORDER BY embedding <=> $1::vector
			LIMIT 5`,
			vector,
		)

		if err != nil {
			log.Println("Database search error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		defer rows.Close()

		results := []SearchResult{}

		for rows.Next() {
			var result SearchResult

			if err := rows.Scan(
				&result.ID,
				&result.Content,
				&result.Similarity,
			); err != nil {
				log.Println("Row scan error:", err)

				c.JSON(http.StatusInternalServerError, gin.H{
					"error": err.Error(),
				})
				return
			}

			results = append(results, result)
		}

		if err := rows.Err(); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"results": results,
		})
	})

	router.POST("/v1/ai/rag", func(c *gin.Context) {
		var req RAGRequest

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": err.Error(),
			})
			return
		}

		topK := req.TopK

		if topK <= 0 {
			topK = 5
		}

		if topK > 20 {
			topK = 20
		}

		threshold := req.Threshold

		if threshold <= 0 {
			threshold = 0.3
		}

		embedding, err := createEmbedding(
			c.Request.Context(),
			client,
			req.Query,
		)

		if err != nil {
			log.Println("Embedding error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		vector := "["

		for i, value := range embedding {
			if i > 0 {
				vector += ","
			}

			vector += strconv.FormatFloat(
				value,
				'f',
				-1,
				64,
			)
		}

		vector += "]"

		rows, err := db.Query(
			c.Request.Context(),
			`SELECT
			content,
			1 - (embedding <=> $1::vector) AS similarity
		FROM documents
		WHERE 1 - (embedding <=> $1::vector) >= $2
		ORDER BY embedding <=> $1::vector
		LIMIT $3`,
			vector,
			threshold,
			topK,
		)

		if err != nil {
			log.Println("Database search error:", err)

			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		defer rows.Close()

		var contextText string
		resultCount := 0

		for rows.Next() {
			var content string
			var similarity float64

			if err := rows.Scan(
				&content,
				&similarity,
			); err != nil {
				log.Println("Row scan error:", err)

				c.JSON(http.StatusInternalServerError, gin.H{
					"error": err.Error(),
				})
				return
			}

			contextText += "\n---\n" + content
			resultCount++
		}

		if err := rows.Err(); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		if resultCount == 0 {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "no relevant documents found",
			})
			return
		}

		prompt := "Answer the user's question using only the provided context. " +
			"If the answer cannot be found in the context, say that the information " +
			"is not available in the provided context.\n\n" +
			"Context:\n" +
			contextText +
			"\n\nQuestion:\n" +
			req.Query

		response, err := client.Chat.Completions.New(
			c.Request.Context(),
			openai.ChatCompletionNewParams{
				Model: openai.ChatModelGPT4o,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage(
						"You are ForgeAI, a helpful RAG assistant. " +
							"Answer questions using only the supplied context. " +
							"Do not invent information.",
					),
					openai.UserMessage(prompt),
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

		answer := response.Choices[0].Message.Content

		c.JSON(http.StatusOK, gin.H{
			"answer":      answer,
			"chunks_used": resultCount,
			"top_k":       topK,
			"threshold":   threshold,
		})
	})

	log.Println("ForgeAI server running on http://localhost:8080")

	if err := router.Run(":8080"); err != nil {
		log.Fatal(err)
	}
}
