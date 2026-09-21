package database

import (
	"context"
	"fmt"

	"forge-ai/internal/config"

	"github.com/jackc/pgx/v5"
)

func New(cfg config.Config) (*pgx.Conn, error) {
	dsn := fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s?sslmode=%s",
		cfg.DBUser,
		cfg.DBPassword,
		cfg.DBHost,
		cfg.DBPort,
		cfg.DBName,
		cfg.DBSSLMode,
	)

	dbConfig, err := pgx.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}

	db, err := pgx.ConnectConfig(
		context.Background(),
		dbConfig,
	)
	if err != nil {
		return nil, err
	}

	if err := db.Ping(context.Background()); err != nil {
		db.Close(context.Background())
		return nil, err
	}

	return db, nil
}
