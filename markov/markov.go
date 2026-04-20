package markov

import (
	"math/rand"
	"time"
)

// Chain representa o motor gerador probabilístico.
// Vai mapear {PalavraAtualID: [Proxima1ID, Proxima2ID, Proxima1ID]}
type Chain struct {
	transitions map[int][]int
	rng         *rand.Rand
}

// New inicializa uma Cadeia de Markov em branco
func New() *Chain {
	// Cria um gerador aleatório focado no relógio atual (para não gerar coisas repetidas toda vez)
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	return &Chain{
		transitions: make(map[int][]int),
		rng:         r,
	}
}

// Train recebe uma sequência gigantesca de Tokens (Ints) da nossa base 
// e correlaciona quem sempre anda de mão dada com quem.
func (c *Chain) Train(tokens []int) {
	if len(tokens) < 2 {
		return // Texto muito curto para aprender padrões lógicos
	}

	for i := 0; i < len(tokens)-1; i++ {
		currentWordID := tokens[i]
		nextWordID := tokens[i+1]

		// Anota que a próxima palavra foi vista após a palavra atual.
		// (Se ocorrer repetição, vai adicionar igual, aumentando sua % de "peso" sorteio).
		c.transitions[currentWordID] = append(c.transitions[currentWordID], nextWordID)
	}
}

// Generate pede para a inteligência adivinhar as "T" próximas palavras com base nos padrões.
func (c *Chain) Generate(seedID int, maxLen int) []int {
	if maxLen <= 0 {
		return nil
	}

	var out []int
	current := seedID
	out = append(out, current)

	for i := 1; i < maxLen; i++ {
		
		// 1. Resgata a lista de possibilidades que conhecemos seguidas a essa palava
		possibles, ok := c.transitions[current]
		
		if !ok || len(possibles) == 0 {
			// Beco sem saída cognitivo. 
			// A IA não domina nenhuma palavra que venha após essa, então ela encerra a frase.
			break
		}

		// 2. Rolagem de Dados de Pesos. Sorteia um índice da "gaveta" de possibilidades
		idx := c.rng.Intn(len(possibles))
		next := possibles[idx]
		
		// 3. Incrementa na frase resultante
		out = append(out, next)
		
		// 4. E a nova atual se torna a recém-sorteada.
		current = next
	}

	return out
}
