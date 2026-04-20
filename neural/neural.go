package neural

import (
	"errors"
	"math/rand"
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
	
	// Pra cada neurônio isolado da nossa camada recém-nascida...
	for i := 0; i < l.Size; i++ { 
		var sum float64
		// ...ele processa o pacote de dados do array
		for j := 0; j < len(inputs); j++ {
			// A ESSÊNCIA DA I.A AQUI: Acúmulo = soma de (Valor da Informação * Peso e Importância dela nesta Sinápse)
			sum += inputs[j] * l.Weights[i][j]
		}
		// Finaliza a dedução: Soma o "Viés" base da rede e filtra ativando pelo ReLU!
		outputs[i] = ReLU(sum + l.Biases[i])
	}

	return outputs, nil
}
