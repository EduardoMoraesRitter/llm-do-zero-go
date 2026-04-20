package tokenizer

import (
	"regexp"
	"strings"
)

// Tokenizer é responsável por converter texto em números e vice-versa
type Tokenizer struct {
	vocab     map[string]int
	inverse   map[int]string
	vocabSize int
	// regex vai definir o padrao de extração: palavras completas ou apenas pontuação (ignorando espaços)
	pattern   *regexp.Regexp 
}

// New inicializa uma nova estrutura de Tokenizer
func New() *Tokenizer {
	return &Tokenizer{
		vocab:   make(map[string]int),
		inverse: make(map[int]string),
		// \p{L}+  -> Captura qualquer bloco de letras do alfabeto (incluindo acentuadas, ex: "não")
		// \p{N}+  -> Captura qualquer bloco de números (ex: "123")
		// [^\p{L}\p{N}\s] -> Captura qualquer caracter que não seja letra, numero, ou espaço (pontuações)
		pattern: regexp.MustCompile(`\p{L}+|\p{N}+|[^\p{L}\p{N}\s]`),
	}
}

// extractTokens é o nosso método mágico de inteligência para quebrar as frases nativamente
func (t *Tokenizer) extractTokens(text string) []string {
	// Padroniza as letras em minúsculas (para que "O" maiúsculo seja o mesmo token que "o" minúsculo)
	text = strings.ToLower(text)
	return t.pattern.FindAllString(text, -1)
}

// Fit converte as string num array de tokens base e constrói o vocabulário
func (t *Tokenizer) Fit(text string) {
	tokens := t.extractTokens(text)

	for _, token := range tokens {
		if _, exists := t.vocab[token]; !exists {
			t.vocab[token] = t.vocabSize
			t.inverse[t.vocabSize] = token
			t.vocabSize++
		}
	}
}

// Encode pega uma frase escrita e devolve a array matemática
func (t *Tokenizer) Encode(text string) []int {
	tokens := t.extractTokens(text)
	var encoded []int

	for _, token := range tokens {
		if id, exists := t.vocab[token]; exists {
			encoded = append(encoded, id)
		} else {
			// Token não foi aprendido: ID -1
			encoded = append(encoded, -1)
		}
	}
	return encoded
}

// Decode pega matriz numérica e converte de volta em strings
func (t *Tokenizer) Decode(ids []int) string {
	var words []string
	for _, id := range ids {
		if token, exists := t.inverse[id]; exists {
			words = append(words, token)
		} else {
			words = append(words, "<UNK>")
		}
	}
	// Junta os tokens. (Num modelo mais complexo de BPE, trataríamos os espaços antes das pontuações,
	// mas para nossa base, devolveremos separados ex: "olá , mundo !").
	return strings.Join(words, " ")
}

// GetVocabSize retorna total de palavras/pontuações que o vocabulário tem
func (t *Tokenizer) GetVocabSize() int {
	return t.vocabSize
}
