package ai

func ChunkText(
	text string,
	chunkSize int,
	overlap int,
) []string {
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

		chunks = append(
			chunks,
			text[start:end],
		)

		if end == len(text) {
			break
		}

		start = end - overlap
	}

	return chunks
}
