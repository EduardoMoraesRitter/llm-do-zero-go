package main

import (
	"fmt"
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
	fmt.Printf("Decode de [%s]: %q (Note 'rainha' não visto no texto base virando <UNK>!)\n", textoTesteP1, tk.Decode(arrayTesteP1))


	fmt.Println("\n==================================================")
	fmt.Println("============ FASE 2: PREVISÃO (MARKOV) ==========")
	fmt.Println("==================================================")
	treinamentoNumeric := tk.Encode(textoTreino)
	mk := markov.New()
	mk.Train(treinamentoNumeric)

	sementeID := tk.Encode("o")[0]
	IdsIA := mk.Generate(sementeID, 12)
	fmt.Printf("[ALUCINAÇÃO DO AI]: 👉 %q\n", tk.Decode(IdsIA))


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
	
	fmt.Printf("[PERGUNTA DA IA] Que palavra lembra os índices [0.8, 0.85, 0.0] do meu Input misterioso?\n")
	melhorMatch, porcentagem, _ := search.NearestNeighbor(vetorMisterioso, memoriaDimensional)
	fmt.Printf("🤖>> A palavra escolhida foi: '%s' (Bate %.2f%% da Similaridade Cosseno no mapa espacial)\n", melhorMatch, porcentagem*100)


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 5: REDE NEURAL (PERCEPTRON) =======")
	fmt.Println("==================================================")
	
	// Vamos dar o Embedding do REI [0.9 0.9 0.1] para ela pensar o que fazer dele
	inputCamada := []float64{0.9, 0.9, 0.1}

	// Criamos a nossa primeira minúscula massa cinzenta
	// Trata-se de uma "Camada Escondida" capaz de aceitar 3 números de Input por palavra
	// E nós decidimos dar o tamanho físico a ela de APENAS 2 Neurônios pensatores (Output)
	redeNeuralArtificial := neural.NewLayer(3, 2)

	fmt.Printf("[INFO] Inputs brutos chegando no circuito de %d neurônios recém criados...\n", redeNeuralArtificial.Size)
	
	outputPensamento, _ := redeNeuralArtificial.Forward(inputCamada)
	
	fmt.Printf("\n🤖>> O Reflexo matemático final da rede gerou os valores lógicos: %v\n", outputPensamento)
	fmt.Println("\n(Observe que '0' indica que o cérebro freiou negativamente a dedução desse neurônio, a clássica ação do limitador de filtro ReLU!)")
	fmt.Println("==================================================")
}
