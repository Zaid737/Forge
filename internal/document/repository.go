package document

import (
	"context"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
)

type Repository struct {
	db *pgx.Conn
}

func NewRepository(db *pgx.Conn) *Repository {
	return &Repository{
		db: db,
	}
}

func vectorString(embedding []float64) string {
	values := make([]string, len(embedding))

	for i, value := range embedding {
		values[i] = strconv.FormatFloat(
			value,
			'f',
			-1,
			64,
		)
	}

	return "[" + strings.Join(values, ",") + "]"
}

func (r *Repository) Create(
	ctx context.Context,
	content string,
	embedding []float64,
) (int64, error) {
	vector := vectorString(embedding)

	var id int64

	err := r.db.QueryRow(
		ctx,
		`INSERT INTO documents (content, embedding)
		 VALUES ($1, $2::vector)
		 RETURNING id`,
		content,
		vector,
	).Scan(&id)

	return id, err
}

func (r *Repository) CreateMany(
	ctx context.Context,
	documents []Document,
	embeddings [][]float64,
) ([]int64, error) {
	tx, err := r.db.Begin(ctx)
	if err != nil {
		return nil, err
	}

	defer tx.Rollback(ctx)

	ids := make([]int64, 0, len(documents))

	for i, doc := range documents {
		vector := vectorString(embeddings[i])

		var id int64

		err := tx.QueryRow(
			ctx,
			`INSERT INTO documents (content, embedding)
			 VALUES ($1, $2::vector)
			 RETURNING id`,
			doc.Content,
			vector,
		).Scan(&id)

		if err != nil {
			return nil, err
		}

		ids = append(ids, id)
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}

	return ids, nil
}

func (r *Repository) Search(
	ctx context.Context,
	embedding []float64,
	limit int,
) ([]SearchResult, error) {
	vector := vectorString(embedding)

	rows, err := r.db.Query(
		ctx,
		`SELECT
			id,
			content,
			1 - (embedding <=> $1::vector) AS similarity
		FROM documents
		ORDER BY embedding <=> $1::vector
		LIMIT $2`,
		vector,
		limit,
	)

	if err != nil {
		return nil, err
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
			return nil, err
		}

		results = append(results, result)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}

func (r *Repository) SearchForRAG(
	ctx context.Context,
	embedding []float64,
	limit int,
	threshold float64,
) ([]SearchResult, error) {
	vector := vectorString(embedding)

	rows, err := r.db.Query(
		ctx,
		`SELECT
			id,
			content,
			1 - (embedding <=> $1::vector) AS similarity
		FROM documents
		WHERE 1 - (embedding <=> $1::vector) >= $2
		ORDER BY embedding <=> $1::vector
		LIMIT $3`,
		vector,
		threshold,
		limit,
	)

	if err != nil {
		return nil, err
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
			return nil, err
		}

		results = append(results, result)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}
