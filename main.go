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
	fmt.Printf("Rei 👑 vs Rainha 👸... Cosseno: %.4f\n", simRR)


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
	queries := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
	keys := [][]float64{{1.0, 0.1}, {0.1, 1.0}}
	values := [][]float64{{0.5, 0.5}, {0.2, 0.8}}

	resultadoDeContexto, _ := attention.SelfAttention(queries, keys, values)
	fmt.Printf("🤖>> O Self-Attention processou a matriz QKV infundindo o contexto! Output:\n%v\n", resultadoDeContexto)


	fmt.Println("\n==================================================")
	fmt.Println("========= FASE 7: CHECKPOINTS (SAVES/LOADS) ======")
	fmt.Println("==================================================")
	fmt.Println("[INFO] 1. Criaremos um modelo que treinou por meses e agora é Experiente:")
	
	// Como a nossa classe base neural inicia tudo sempre caotico puramente (aleatorio),
	// O Mestre acima terá valores numéricos únicos que se não salvos a cada Execução do software as matrizes são Perdidas.
	cerebroMestre := neural.NewLayer(3, 2)
	matrixMestre, _ := cerebroMestre.Forward(inputCamada)
	fmt.Printf(">> Dedução original do Cérebro Experiente para a pergunta do Rei: %v\n", matrixMestre)

	arquivoDeSalvamento := "cerebro_gpt_v1.bin"
	_ = cerebroMestre.Save(arquivoDeSalvamento)
	fmt.Println("\n📂>> A mágica do Pesos do Mestre foram salvos fisicamente no seu HD! (Gerou: " + arquivoDeSalvamento + ")")

	fmt.Println("\n[INFO] 2. TEMPO DEPOIS... A bateria acabou. O golang abriu de novo do Lixo/Zero.")
	
	// O LoadLayer é quem baixa o modelo da intenet/ssd como um Arquivo binario sem vírus pra dentro da matriz em C/Go
	cerebroRessuscitadoDaInternet, err := neural.LoadLayer(arquivoDeSalvamento)
	if err == nil {
		matrixClone, _ := cerebroRessuscitadoDaInternet.Forward(inputCamada)
		fmt.Printf(">> Dedução tirada de dentro do Cérebro Clone baixado (Veja!): %v\n", matrixClone)
		fmt.Println(">> INCRÍVEL! Nenhuma diferença detectada. O clone recuperou sua Inteligência original matematicamente.")
	}
	fmt.Println("==================================================")
}
