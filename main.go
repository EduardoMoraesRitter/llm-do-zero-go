package main

import (
	"fmt"
	"llm-do-zero/markov"
	"llm-do-zero/tokenizer"
)

func main() {
	fmt.Println("=== 🚀 Fase 2: Motor de Previsão de Texto (Markov) ===")

	// Texto grande para dar algum "repertório" estatístico à IA
	textoTreino := `
O rato roeu a roupa do rei de Roma, mas não roeu o relógio . 
O rei zangado mandou o rato para o reino distante .
Longe do rei, o rato roeu muito queijo .
O queijo era bom demais para o rato !
O rei de Roma vestiu outra roupa e esqueceu do rato e do queijo .
	`

	// ==========================================
	// 1. FASE DE TOKENIZATION (Compreensão Numérica)
	// ==========================================
	tk := tokenizer.New()
	tk.Fit(textoTreino)

	fmt.Printf("[Tokenizer] Vocabulário Base absorvido: %d identificadores numéricos criados.\n", tk.GetVocabSize())

	// Convertemos o texto de aprendizado para Matriz de Inteiros:
	treinamentoNumeric := tk.Encode(textoTreino)

	// ==========================================
	// 2. FASE MARKOV (Pesos Probabilísticos)
	// ==========================================
	mk := markov.New()
	mk.Train(treinamentoNumeric)
	fmt.Println("[Markov]    Padrões e probabilidades cruzadas extraídas com sucesso.")

	// ==========================================
	// 3. FASE DE INFERÊNCIA (Gerando texto novo)
	// ==========================================
	fmt.Println("\n--- 🤖 IA Gerando Frases Inéditas ---")

	// Escolhemos uma Semente com a qual ela será forçada a iniciar a frase (ex: "o")
	palavraSemente := "o"
	sementeID := tk.Encode(palavraSemente)[0]

	// Pede ao motor para tentar achar as próximas 15 palavras prováveis em sequência
	IdsIA := mk.Generate(sementeID, 15)

	// O formato é puramente numérico (A linguagem que a IA pensa)
	fmt.Printf("A IA calculou esta matriz:\n%v\n\n", IdsIA)

	// O processo Reverso pra traduzir aos humanos (Decode)
	fraseMágica := tk.Decode(IdsIA)
	
	fmt.Printf("Tradução ao humano:\n👉 %q\n", fraseMágica)
	fmt.Println("\n(Se você rodar o 'go run main.go' dezenas de vezes, ela construirá frases diferentes dadas as probabilidades bifurcadas!).")
}
