package vectors

import (
	"errors"
	"math"
)

// Embedding é a forma "evoluída" dos nossos simples IDs de Tokens da Fase 1.
// Num LLM, cada palavra ou conceito é transformada num array gigantesco de números decimais
// (ex: no ChatGPT-3, um Embedding tem em média 12.288 números) que representam sua posição num plano.
type Embedding []float64

// DotProduct (Produto Escalar) - a base da Álgebra Linear!
// Multiplica os valores dimensionais de duas palavras para começar a tentar achar similaridades de sentido
func DotProduct(v1, v2 Embedding) (float64, error) {
	// Duas palavras só podem ser quantificadas matematicamente na mesma régua/dimensão
	if len(v1) != len(v2) {
		return 0, errors.New("os dois vetores não possuem o mesmo número de dimensões na representação")
	}

	var soma float64
	for i := 0; i < len(v1); i++ {
		soma += v1[i] * v2[i]
	}
	return soma, nil
}

// Magnitude (ou Norma) calcula a "distância geométrica euclidiana em linha reta" de uma palavra
// desde o ponto de origem [0, 0] no espaço multidimensional até ela.
func Magnitude(v Embedding) float64 {
	var soma float64
	for _, val := range v {
		soma += val * val
	}
	return math.Sqrt(soma)
}

// CosineSimilarity é o "Grande Algoritmo de Busca" padrão na indústria de Inteligência Artificial.
// Acha o ângulo (Cosseno) de proximidade visual entre as palavras.
// Retornos:
// -> Próximo de 1.0 = Têm significados ou contextos IDÊNTICOS na língua (ex: Rei e Rainha).
// -> Próximo de 0.0 = Não têm nada a ver um com o outro fisicamente/ideologicamente.
// -> Próximo de -1.0 = São o extremo oposto (ex: Bem e Mal, Claro e Escuro).
func CosineSimilarity(v1, v2 Embedding) (float64, error) {
	dot, err := DotProduct(v1, v2)
	if err != nil {
		return 0, err
	}

	magA := Magnitude(v1)
	magB := Magnitude(v2)

	// Cuidado matemático (Não pode dividir nada por absoluto zero)
	if magA == 0 || magB == 0 {
		return 0, errors.New("vetor morto com valor matemático zero")
	}

	return dot / (magA * magB), nil
}
