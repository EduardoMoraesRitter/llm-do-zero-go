package search

import (
	"errors"
	"llm-do-zero/vectors"
)

// NearestNeighbor (Busca por Vizinho Mais Próximo).
// Essa função é exatamente o coração de sistemas como RAG (Retrieval-Augmented Generation) ou buscas no Google.
// Passamos um vetor mágico (target) que queremos entender e varremos todo o "banco de memórias matemáticas" 
// da IA tentando achar o elemento de maior Similaridade de Cosseno com ele.
func NearestNeighbor(target vectors.Embedding, db map[string]vectors.Embedding) (string, float64, error) {
	if len(db) == 0 {
		return "", 0, errors.New("o banco de dados da inteligência está vazio")
	}

	melhorPalavra := ""
	melhorScore := -2.0 // Iniciamos abaixo de -1 (o pior cosseno existente)

	for palavra, embedding := range db {
		// Puxamos a fórmula que construímos na Fase 3
		score, err := vectors.CosineSimilarity(target, embedding)
		if err != nil {
			continue // Pula vetores quebrados
		}

		// Matemática pura salvando o dia: Se a similaridade for a maior já vista, salvamos quem venceu!
		if score > melhorScore {
			melhorScore = score
			melhorPalavra = palavra
		}
	}

	return melhorPalavra, melhorScore, nil
}
