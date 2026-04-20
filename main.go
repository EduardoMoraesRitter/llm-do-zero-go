package main

import (
	"fmt"
	"llm-do-zero/markov"
	"llm-do-zero/search"
	"llm-do-zero/tokenizer"
	"llm-do-zero/vectors"
)

func main() {
	textoTreino := `
O rato roeu a roupa do rei de Roma, mas não roeu o relógio ! 
O rei zangado mandou o rato para o reino distante .
Longe do rei, o rato roeu muito queijo .
O queijo era bom demais para o rato !
O rei de Roma vestiu outra roupa e esqueceu do rato e do queijo .
	`

	fmt.Println("==================================================")
	fmt.Println("============ FASE 1: O TOKENIZER ===============")
	fmt.Println("==================================================")
	tk := tokenizer.New()
	tk.Fit(textoTreino)
	
	fmt.Printf("[INFO] O Tokenizer analisou o texto bruto e criou %d tokens únicos em seu dicionário numérico.\n", tk.GetVocabSize())
	
	textoTesteP1 := "o rato e rainha"
	fmt.Printf("\nTeste Prático (Transformando '%s' num array): \n", textoTesteP1)
	arrayTesteP1 := tk.Encode(textoTesteP1)
	fmt.Printf("Matriz Encode: %v\n", arrayTesteP1)
	fmt.Printf("Matriz Decode: %q (Prova de que a palavra 'rainha' era desconhecida pelo modelo e virou ID -1)\n", tk.Decode(arrayTesteP1))


	fmt.Println("\n==================================================")
	fmt.Println("============ FASE 2: PREVISÃO (MARKOV) ==========")
	fmt.Println("==================================================")
	treinamentoNumeric := tk.Encode(textoTreino)
	mk := markov.New()
	mk.Train(treinamentoNumeric)

	sementeID := tk.Encode("o")[0]
	IdsIA := mk.Generate(sementeID, 12)
	
	fmt.Printf("[INFO] Padrões e correlações estatísticas extraídas.\n")
	fmt.Printf("\nResultado: A IA 'adivinhando' frases lógicas via probabilidade!\n")
	fmt.Printf("Tradução ao humano da Cadeia (Seed = 'o'): 👉 %q\n", tk.Decode(IdsIA))


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 3: SIMILARIDADE (VECTORES) ========")
	fmt.Println("==================================================")
	// Vetores dimensionais baseados em [Riqueza, Realeza, Rato-nível]
	vetorRei := vectors.Embedding{0.9, 0.9, 0.1}
	vetorRainha := vectors.Embedding{0.9, 0.9, 0.0}
	vetorRato := vectors.Embedding{0.1, 0.0, 0.9}

	fmt.Println("[INFO] Testando a fórmula matemática do Cosseno dentro dos tensores de palavras.")
	simRR, _ := vectors.CosineSimilarity(vetorRei, vetorRainha)
	fmt.Printf("Rei 👑 vs Rainha 👸...\n>> Cosseno: %f\n", simRR)

	simRM, _ := vectors.CosineSimilarity(vetorRei, vetorRato)
	fmt.Printf("\nRei 👑 vs Rato 🐭...\n>> Cosseno: %f\n", simRM)


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 4: BUSCA SEMÂNTICA (SEARCH) =======")
	fmt.Println("==================================================")

	// Aqui a IA tem a sua "Base de Dados Vetorial" viva!
	memoriaDimensional := map[string]vectors.Embedding{
		"Rei":    vetorRei,
		"Rainha": vetorRainha,
		"Rato":   vetorRato,
	}

	// Criaremos um vetor misterioso que representa: "Alta Riqueza (0.8), Alta Realeza (0.85), Nem um pouco roedor (0.0)"
	// Em um modelo real, a IA plotou um ponto tridimensional quando leu a palavra "Príncipe" num livro ou na nossa pergunta de chat.
	vetorMisterioso := vectors.Embedding{0.8, 0.85, 0.0}

	fmt.Println("[INFO] Consultando a Base Vectorial: [Rei, Rainha, Rato]")
	fmt.Println("[PERGUNTA DA IA] Qual palavra no meu cérebro matemático melhor se encaixa neste vetor misterioso (Alto Poder Real)?")
	
	melhorMatch, porcentagem, _ := search.NearestNeighbor(vetorMisterioso, memoriaDimensional)
	
	fmt.Printf("\n🤖>> Resposta Final: A palavra que mais faz sentido é a '%s' (Semelhança Cosseno: %.2f%%)\n", melhorMatch, porcentagem*100)
	fmt.Println("==================================================")
}
