package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"forge-ai/internal/ai"
	"forge-ai/internal/config"
	"forge-ai/internal/database"
	"forge-ai/internal/document"
	"forge-ai/internal/handler"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}

	db, err := database.New(cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close(context.Background())

	openAIClient := openai.NewClient(
		option.WithAPIKey(cfg.OpenAIAPIKey),
	)

	aiService := ai.NewService(openAIClient)

	documentRepository := document.NewRepository(db)

	documentService := document.NewService(
		documentRepository,
		aiService,
	)

	aiHandler := handler.NewAIHandler(
		aiService,
		documentService,
	)

	healthHandler := handler.NewHealthHandler(db)

	router := gin.Default()

	router.GET("/health", healthHandler.Health)

	router.POST("/v1/ai/chat", aiHandler.Chat)
	router.POST("/v1/ai/extract", aiHandler.Extract)
	router.POST("/v1/ai/embed", aiHandler.Embed)

	router.POST("/v1/ai/documents", aiHandler.CreateDocument)
	router.POST("/v1/ai/search", aiHandler.Search)
	router.POST("/v1/ai/rag", aiHandler.RAG)

	router.POST("/v1/ai/tools", aiHandler.Tools)

	server := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		log.Printf(
			"ForgeAI server running on http://localhost:%s",
			cfg.Port,
		)

		if err := server.ListenAndServe(); err != nil &&
			!errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()

	stop := make(chan os.Signal, 1)

	signal.Notify(
		stop,
		os.Interrupt,
		syscall.SIGTERM,
	)

	<-stop

	log.Println("Shutting down ForgeAI...")

	ctx, cancel := context.WithTimeout(
		context.Background(),
		10*time.Second,
	)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	log.Println("ForgeAI stopped")
}
