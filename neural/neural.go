package neural

import (
	"encoding/gob"
	"errors"
	"math/rand"
	"os"
	"time"
)

// Layer representa uma camada densa (Linear/Fully Connected) clássica de neurônios
type Layer struct {
	Weights [][]float64 // Matriz de pesos que o neurônio dá a cada input: [Neurônio_ID][Input_Feature_ID]
	Biases  []float64   // Viés (Bias) para cada neurônio
	Size    int         // Quantidade de neurônios na camada para formar nosso cérebro artificial
}

// NewLayer "dá a luz" à uma camada de neurônios inicializando pesos vazios de forma estocástica (aleatórios)
func NewLayer(inputSize int, outputSize int) *Layer {
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src)

	weights := make([][]float64, outputSize)
	biases := make([]float64, outputSize)

	for i := 0; i < outputSize; i++ {
		weights[i] = make([]float64, inputSize)
		// Simula inicialização padrão de grandes redes de IA em seus neurônios
		for j := 0; j < inputSize; j++ {
			weights[i][j] = r.Float64()*0.4 - 0.2 // Valor que varia de -0.2 a 0.2
		}
		biases[i] = r.Float64()*0.4 - 0.2
	}

	return &Layer{
		Weights: weights,
		Biases:  biases,
		Size:    outputSize,
	}
}

// ReLU (Rectified Linear Unit). O pilar base da Inteligência Artificial Profunda (Deep Learning).
// Corta deduduzões negativas: Se a sinapse for menor que zero, a ativação simplesmente trava a porta (Retorna 0).
func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Forward é a Função onde o neurônio realmente liga e "Pensa" (Passo-à-frente).
// Recebe os estímulos do Input (nossos arrays/embeddings), multiplica por seus 'Pesos' e passa na Ativação.
func (l *Layer) Forward(inputs []float64) ([]float64, error) {
	if len(inputs) == 0 || len(inputs) != len(l.Weights[0]) {
		return nil, errors.New("a entrada informada não se encaixa nos conectores sinápticos físicos da nossa camada neural desenvolvida")
	}

	outputs := make([]float64, l.Size)
	
	for i := 0; i < l.Size; i++ { 
		var sum float64
		for j := 0; j < len(inputs); j++ {
			sum += inputs[j] * l.Weights[i][j]
		}
		outputs[i] = ReLU(sum + l.Biases[i])
	}

	return outputs, nil
}

// Save (O Checkpoint) pega o estado biológico digital atual (As matrizes de Weights aprendidos e Biases)
// e compila/empacota como estático (arquivo '.bin') em disco. É a cópia exata comportamental dos arquivos .safetensors e .gguf.
func (l *Layer) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Encoder embutido do Golang purifica as memórias pra bits crus muito leves.
	encoder := gob.NewEncoder(file)
	return encoder.Encode(l)
}

// LoadLayer pega o pacote misterioso e offline do seu HD (A alma da Inteligência Matemática) e injeta
// reescrevendo ativamente a Inteligência Aleatória pro modo do gênio que você Treinou no dia anterior.
func LoadLayer(filename string) (*Layer, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var camadaRessuscitada Layer
	decoder := gob.NewDecoder(file)
	
	if err := decoder.Decode(&camadaRessuscitada); err != nil {
		return nil, err
	}
	return &camadaRessuscitada, nil
}
