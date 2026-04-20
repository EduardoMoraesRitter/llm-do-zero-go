package main

import (
	"fmt"
	"llm-do-zero/tokenizer"
)

func main() {
	fmt.Println("=== 🚀 Testando o Tokenizer (Versão com Expressões Regulares) ===")

	textoTreino := "O rato roeu a roupa do rei de Roma, mas não roeu o relógio!"
	fmt.Printf("Texto de Treino:\n%q\n\n", textoTreino)

	// 1. Inicializa e Constroi Vocabulário
	tk := tokenizer.New()
	tk.Fit(textoTreino)

	fmt.Printf("Vocabulário gerado: %d tokens únicos criados.\n", tk.GetVocabSize())

	// 2. Testando o Encode
	// Note o uso de pontuação e maiúsculas misturadas - ele agora lida com tudo inteligentemente.
	textoTeste := "O rei, a rainha e a coroa!"

	fmt.Printf("\n--- Codificando Texto (Encode) ---\n")
	fmt.Printf("Texto de Teste: %q\n", textoTeste)

	codificado := tk.Encode(textoTeste)
	fmt.Printf("Matriz gerada: %v\n", codificado)

	// 3. Testando o Decode
	fmt.Printf("\n--- Decodificando Texto (Decode) ---\n")
	decodificado := tk.Decode(codificado)
	fmt.Printf("Resultado: %q\n", decodificado)
}
