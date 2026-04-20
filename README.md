# LLM do Zero em Go 🚀

Este é um projeto educacional focado em construir as bases de um Large Language Model (LLM) do zero, utilizando a linguagem de programação Go (Golang). A ideia é desmistificar o funcionamento interno dos modelos de IA, explorando os conceitos de NLP e Deep Learning na prática, sem bibliotecas pesadas de machine learning.

## 📂 Estrutura e Módulos do Projeto

Para melhor aprendizado, a arquitetura foi dividida em diferentes pacotes que representam as principais etapas da geração e compreensão de linguagem por IAs:

* **`tokenizer/`**: Lida com a transformação da linguagem humana em matrizes matemáticas. É responsável por separar os textos e palavras na granularidade correta para gerar um vocabulário indexado e numérico.
* **`vectors/`**: Base da álgebra linear para o projeto. Inclui o tratamento das representações das palavras (*Word Embeddings*) de modo a inseri-las num Espaço Vetorial para representar significados.
* **`search/`**: Lida com algoritmos de agrupamento ou pesquisas usando a proximidade entre os vetores das palavras (como a *Cosine Similarity*) para avaliar similaridades de contexto.
* **`markov/`**: Implementa Cadeias de Markov, que exploram predição da próxima palavra de um texto somente analisando suas distribuições de probabilidade estatística histórica (uma ótima baseline antes de entrar nas redes neurais).
* **`neural/`**: Aborda os fundamentos das Redes Neurais clássicas (camadas *Feed Forward*, *Perceptrons* e funções de ativação do modelo).
* **`attention/`**: O núcleo de processamento dos modelos de hoje (Transformers). Implementa funções para *Self-Attention* (Autoatenção) e *Multi-Head Attention*, avaliando não só as palavras, mas toda a correlação entre elas em uma frase inteira.

## 🛠️ Como Executar

**Pré-requisitos gerais:**
* [Go (Golang)](https://go.dev/) devidamente instalado na máquina.

**Passo a Passo:**
Acesse este diretório via terminal de sua preferência (`cd llm-do-zero`) de onde quer que você tenha baixado.
Basta então chamar o compilador na raiz por meio do comando `run`:

```bash
go run main.go
```

## 🎯 Próximos Passos
- [x] **Fase 1:** Implementar o *Tokenizer* focado em RegEx (ignorando maiúsculas e modelando pontuações). *(Concluído)*
- [ ] **Fase 2:** Construir o primeiro modelo gerador básico na pasta `markov`.
- [ ] **Fase 3:** Construir os vetores numéricos de similaridade na pasta `vectors`.
- [ ] ... (A evoluir durante a construção)

---

## 🏁 Histórico de Progresso

### Fase 1: O Módulo Tokenizer (Concluído)
Criamos um sistema avançado de geração de vocabulário no pacote `tokenizer/`. Ele é capaz de ler textos de treinamento, separar e compreender individualmente as palavras e até as pontuações de forma independente usando Expressões Regulares (`RegEx` via `\p{L}`). Ignorando a diferença entre maiúsculas e minúsculas, nosso Tokenizer distribui identificadores numéricos e executa 3 passos centrais da Inteligência Artificial:
* **Fit (Treino)**: Aprende e varre os textos base, construindo um dicionário unificado (uma palavra = um ID `int`).
* **Encode**: Traduz textos humanos para o modelo matriz matemático `[]int`. Palavras não vistas no treino são batizadas inteligentemente com `ID -1`.
* **Decode**: Executa o reverso. Traduz a matriz lógica novamente para frases normais legíveis. Palavras desconhecidas na matriz que possuíam peso -1 se tornam `<UNK>` (Unknown).
