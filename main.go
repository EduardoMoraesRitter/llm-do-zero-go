package main

import (
	"fmt"
	"llm-do-zero/markov"
	"llm-do-zero/tokenizer"
	"llm-do-zero/vectors"
)

func main() {
	fmt.Println("============ FASE 2: PREVISÃO (MARKOV) ============")
	textoTreino := `
O rato roeu a roupa do rei de Roma, mas não roeu o relógio . 
O rei zangado mandou o rato para o reino distante .
Longe do rei, o rato roeu muito queijo .
O queijo era bom demais para o rato !
O rei de Roma vestiu outra roupa e esqueceu do rato e do queijo .
	`

	tk := tokenizer.New()
	tk.Fit(textoTreino)
	treinamentoNumeric := tk.Encode(textoTreino)

	mk := markov.New()
	mk.Train(treinamentoNumeric)

	palavraSemente := "o"
	sementeID := tk.Encode(palavraSemente)[0]

	IdsIA := mk.Generate(sementeID, 12)
	fraseMágica := tk.Decode(IdsIA)
	
	fmt.Printf("Tradução ao humano da Cadeia (Seed = 'o'): 👉 %q\n", fraseMágica)


	fmt.Println("\n\n============ FASE 3: SIMILARIDADE SEMÂNTICA (VECTORES) ============")
	
	// Num projeto real, esses super-arrays decimais seriam preenchidos automaticamente
	// pela rede neural em tempo real após mastigar a Wikipédia inteira, agrupando
	// as coisas no plano dimensional. Aqui estamos "mockando" (forçando na mão) o cérebro
	// do LLM pra entender o peso tridimensional de "poder/riqueza", "realeza" e "roedor".
	
	// Rei = Rico (0.9), Realeza Absoluta (0.9), Nivel Roedor Mínimo (0.1)
	vetorRei := vectors.Embedding{0.9, 0.9, 0.1}
	
	// Rainha = Rica (0.9), Realeza Absoluta (0.9), Nivel Roedor Mínimo (0.0)
	vetorRainha := vectors.Embedding{0.9, 0.9, 0.0}
	
	// Rato = Pouco rico (0.1), Zero realeza (0.0), Alto nível de Roedor (0.9)
	vetorRato := vectors.Embedding{0.1, 0.0, 0.9}

	fmt.Println("Testando familiaridade semântica matemática entre 'Rei' 👑 e 'Rainha' 👸...")
	simRR, _ := vectors.CosineSimilarity(vetorRei, vetorRainha)
	fmt.Printf(">> Grau de Similaridade: %f (Próximo de 1.0 é extremo parentesco semântico!)\n", simRR)

	fmt.Println("\nTestando familiaridade semântica matemática entre 'Rei' 👑 e 'Rato' 🐭...")
	simRM, _ := vectors.CosineSimilarity(vetorRei, vetorRato)
	fmt.Printf(">> Grau de Similaridade: %f (Próximo ou Menor que 0.0 significa que não compartilham quase nenhum sentido)\n", simRM)
}
