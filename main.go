package main

import (
	"fmt"
	"llm-do-zero/attention"
	"llm-do-zero/markov"
	"llm-do-zero/neural"
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
	
	textoTesteP1 := "o rato e rainha"
	arrayTesteP1 := tk.Encode(textoTesteP1)
	fmt.Printf("[INFO] O Tokenizer analisou o texto bruto e criou %d tokens únicos vocabulário.\n", tk.GetVocabSize())
	fmt.Printf("Decode de [%s]: %q\n", textoTesteP1, tk.Decode(arrayTesteP1))


	fmt.Println("\n==================================================")
	fmt.Println("============ FASE 2: PREVISÃO (MARKOV) ==========")
	fmt.Println("==================================================")
	treinamentoNumeric := tk.Encode(textoTreino)
	mk := markov.New()
	mk.Train(treinamentoNumeric)

	sementeID := tk.Encode("o")[0]
	fmt.Printf("[ALUCINAÇÃO DO AI]: 👉 %q\n", tk.Decode(mk.Generate(sementeID, 12)))


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 3: SIMILARIDADE (VECTORES) ========")
	fmt.Println("==================================================")
	vetorRei := vectors.Embedding{0.9, 0.9, 0.1}
	vetorRainha := vectors.Embedding{0.9, 0.9, 0.0}
	vetorRato := vectors.Embedding{0.1, 0.0, 0.9}

	simRR, _ := vectors.CosineSimilarity(vetorRei, vetorRainha)
	simRM, _ := vectors.CosineSimilarity(vetorRei, vetorRato)
	fmt.Printf("Rei 👑 vs Rainha 👸... Cosseno: %.4f\n", simRR)
	fmt.Printf("Rei 👑 vs Rato 🐭... Cosseno: %.4f\n", simRM)


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 4: BUSCA SEMÂNTICA (SEARCH) =======")
	fmt.Println("==================================================")
	memoriaDimensional := map[string]vectors.Embedding{"Rei": vetorRei, "Rainha": vetorRainha, "Rato": vetorRato}
	vetorMisterioso := vectors.Embedding{0.8, 0.85, 0.0}
	
	melhorMatch, porcentagem, _ := search.NearestNeighbor(vetorMisterioso, memoriaDimensional)
	fmt.Printf("🤖>> A palavra escolhida pro Mistério foi: '%s' (Semelhança Cosseno: %.2f%%)\n", melhorMatch, porcentagem*100)


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 5: REDE NEURAL (PERCEPTRON) =======")
	fmt.Println("==================================================")
	inputCamada := []float64{0.9, 0.9, 0.1}
	redeNeuralArtificial := neural.NewLayer(3, 2)
	outputPensamento, _ := redeNeuralArtificial.Forward(inputCamada)
	fmt.Printf("🤖>> O Reflexo matemático final da rede gerou: %v\n", outputPensamento)


	fmt.Println("\n==================================================")
	fmt.Println("========== FASE 6: ATTENTION (O SEGREDO) =========")
	fmt.Println("==================================================")
	
	// Faremos a Simulação do mecanismo de atenção focando apenas um universo minúsculo e visual 
	// onde só temos 3 Palavras numa frase. (Ex: "O"(1), "RATO"(2), "COMEU"(3)). E só 2 traços espaciais de vetor.
	
	// Q (Query) - O que a palavra está procurando achar nas frases que estão do lado
	queries := [][]float64{
		{1.0, 0.0}, // Palavra 1 (Ex: O) percebendo algo
		{0.0, 1.0}, // Palavra 2 (Ex: RATO) percebendo algo
		{1.0, 1.0}, // Palavra 3 (Ex: COMEU) escaneando tudo
	}
	
	// K (Keys)  - O segredo que a palavra "abre" pras vizinhas analisaram
	keys := [][]float64{
		{1.0, 0.1}, 
		{0.1, 1.0}, 
		{1.0, 1.0},
	}
	
	// V (Values) - O significado puro daquela palavra passado finalmente adiante
	values := [][]float64{
		{0.5, 0.5},
		{0.2, 0.8},
		{0.9, 0.9},
	}

	fmt.Println("[TEMPO DE FLUXO] Multiplicando Querie com chaves (Q x K) e enxertando em todos os Valores...")
	
	resultadoDeContexto, _ := attention.SelfAttention(queries, keys, values)

	fmt.Println("\n🤖>> Eis as Novas Representações das palavras depois que leram e compreenderam a frase inteira:")
	for pos, conceitoAmpliado := range resultadoDeContexto {
		fmt.Printf("Palavra %d ganhou as propriedades de Sentido Mútuo: %v\n", pos+1, conceitoAmpliado)
	}
	fmt.Println("==================================================")
}
