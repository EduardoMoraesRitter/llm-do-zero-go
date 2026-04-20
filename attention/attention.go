package attention

import (
	"errors"
	"math"
)

// Softmax transforma uma lista de notas matemáticas cruas (scores) em uma distribuição de probabilidade fina.
// Transforma valores de modo que ao somar todos, tenhamos um exato "1.0" (100% de atenção).
func Softmax(scores []float64) []float64 {
	maxScore := scores[0]
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}

	var sumExp float64
	exps := make([]float64, len(scores))
	for i, s := range scores {
		exps[i] = math.Exp(s - maxScore) // A subtração do max previne que a array estoure o limite de números flutuantes pesados (Estabilidade numérica)
		sumExp += exps[i]
	}

	// Normaliza pra todos dividirem o "bolo" entre si 
	for i := range exps {
		exps[i] /= sumExp 
	}
	return exps
}

// SelfAttention é a joia da coroa arquitetural do paper "Attention Is All You Need" (Do Google em 2017).
// Aqui cada palavra OLHA para TODAS AS OUTRAS da frase de uma vez só, e distribui suas "morgas percentuais" 
// de atenção, ganhando contexto das vizinhas!
func SelfAttention(queries [][]float64, keys [][]float64, values [][]float64) ([][]float64, error) {
	if len(queries) == 0 || len(keys) == 0 || len(values) == 0 {
		return nil, errors.New("matrizes neurais de Query/Key/Value quebradas")
	}

	seqLen := len(queries) // Número de palavras na nossa frase
	dKey := len(keys[0])   // O tamanho dimensões de "Características" de cada palavra

	// Nossa matriz final (Output), após a palavra assimilar o Sentido de todas as outras juntas
	output := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ { // Pra cada Palavra "i" (A Query de Pergunta)
		scores := make([]float64, seqLen)

		// 1. O Computador choca a palavra 'i' contra todas as outras palavras 'j' e nota a pontuação (Dot Product)
		for j := 0; j < seqLen; j++ {
			var dot float64
			for d := 0; d < dKey; d++ {
				dot += queries[i][d] * keys[j][d]
			}
			// Escalamento genial para IA não viciar em valores muito grandes: Score = (Q * K) / Raiz(Dimensão)
			scores[j] = dot / math.Sqrt(float64(dKey))
		}

		// 2. Usando o Softmax, distribui essas pontuações brutas até dar 100%
		attentionWeights := Softmax(scores)

		// 3. Multiplica o valor em '%' das outras palavras e junta pro próprio cérebro (A Absorção)
		dValue := len(values[0])
		contextVector := make([]float64, dValue)
		
		for j := 0; j < seqLen; j++ { // j = de quem vamos sugar a sabedoria (as outras palavras)
			for d := 0; d < dValue; d++ { // Iterando nas propriedades puras delas
				// "Eu incorporo uma fatiazinha da palavra J dependendo de quão alta foi minha ligação neural/ação de Peso com ela!"
				contextVector[d] += attentionWeights[j] * values[j][d]
			}
		}

		output[i] = contextVector // Atualiza O que Diabos a palavra I de fato Significa Agora.
	}

	return output, nil
}
