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

* [x] **Fase 1:** Implementar o *Tokenizer* focado em RegEx (ignorando maiúsculas e modelando pontuações). *(Concluído)*
* [x] **Fase 2:** Construir o primeiro modelo gerador básico na pasta `markov`. *(Concluído)*
* [x] **Fase 3:** Construir os vetores numéricos de similaridade na pasta `vectors`. *(Concluído)*
* [x] **Fase 4:** Criar o mecanismo de Busca Semântica (Nearest Neighbor RAG) na pasta `search`. *(Concluído)*
* [ ] ... (A evoluir durante a construção)

---

## 🏁 Histórico de Progresso

### Fase 1: O Módulo Tokenizer (Concluído)

Criamos um sistema avançado de geração de vocabulário no pacote `tokenizer/`. Ele é capaz de ler textos de treinamento, separar e compreender individualmente as palavras e até as pontuações de forma independente usando Expressões Regulares (`RegEx` via `\p{L}`). Ignorando a diferença entre maiúsculas e minúsculas, nosso Tokenizer distribui identificadores numéricos e executa 3 passos centrais da Inteligência Artificial:

* **Fit (Treino)**: Aprende e varre os textos base, construindo um dicionário unificado (uma palavra = um ID `int`).
* **Encode**: Traduz textos humanos para o modelo matriz matemático `[]int`. Palavras não vistas no treino são batizadas inteligentemente com `ID -1`.
* **Decode**: Executa o reverso. Traduz a matriz lógica novamente para frases normais legíveis. Palavras desconhecidas na matriz que possuíam peso -1 se tornam `<UNK>` (Unknown).

**🤖 Exemplo de Processamento:**

```text
=== 🚀 Testando o Tokenizer (Versão com Expressões Regulares) ===
Texto de Treino:
"O rato roeu a roupa do rei de Roma, mas não roeu o relógio!"

Vocabulário gerado: 14 tokens únicos criados.

--- Codificando Texto (Encode) ---
Texto de Teste: "O rei, a rainha e a coroa!"
Matriz gerada: [0 6 9 3 -1 -1 3 -1 13]

--- Decodificando Texto (Decode) ---
Resultado: "o rei , a <UNK> <UNK> a <UNK> !"
```

---

### Fase 2: O Motor de Geração (Cadeias de Markov) (Concluído)

Integrado logicamente no módulo `markov/`, nosso modelo atua como o primeiro cérebro gerador do projeto. Sem precisar de pesadas redes neurais, ele aprende as transições estatísticas matemáticas enviadas pelo Tokenizer para formular frases próprias "inéditas" baseado pesadamente nas probabilidades de continuação de palavras!

#### 🕵️ O que são as Cadeias de Markov?

* **A Origem do Nome**: Coroa o genial matemático russo **Andrey Markov**.
* **Quando foi criado?**: O conceito (*Markov Chains*) foi provado pela primeira vez em 1906, pasme, quase meia década antes das estruturas computacionais modernas.
* **O que é?**: Trata-se de um modelo estocástico que descreve uma sequência de eventos possíveis onde a probabilidade da ocorrência do "próximo estado" (Ex: a próxima palavra gerada no texto) **depende única e exclusivamente do estado presente**, desprezando as palavras extremamente antigas geradas na frase.
* **Para que serviu na História e neste LLM?**: É o tataravô da previsão temporal. Usado largamente na biologia, nas bolsas de valores e na computação, Markov foi a baseline de sistemas de processamento de linguagens antigas (como os teclados que tentavam prever a próxima palavra da sua mensagem SMS no início dos anos 2000). A máquina memoriza todas as vezes que a palavra "A" é sucedida por "B" ou "C" e lança dados ponderados (no caso do golang, com `rand` com *seeding* temporal) para prever o que o seu robô te responderá!

**🎲 Exemplo de "Alucinação" da Máquina na Fase 2:**

Pedimos para ela iniciar uma frase obrigada a utilizar o ID `0` (Letra *O*)... E em seguida o algoritmo usou estatística pura, conectando as saídas prováveis matematicamente num *array* limpo até montar uma frase assustadoramente parecida com coerência humana:

```text
=== 🚀 Fase 2: Motor de Previsão de Texto (Markov) ===
[Tokenizer] Vocabulário Base absorvido: 30 identificadores numéricos criados.
[Markov]    Padrões e probabilidades cruzadas extraídas com sucesso.

--- 🤖 IA Gerando Frases Inéditas ---
A IA calculou esta matriz de sequência temporal:
[0 1 25 0 1 28 5 6 9 0 1 2 3 4 5]

Tradução final repassada aos humanos (Decode):
👉 "o rato ! o rato e do rei , o rato roeu a roupa do"
```

---

### Fase 3: Vetores de Similaridade e Embeddings (Concluído)

Deixamos o mundo isolado da Fase 1 para trás e entramos no campo contínuo das abstrações dimensionais (Álgebra Linear) no módulo `vectors/`. Num Large Language Model (LLM) da atualidade, é exatamente através deste módulo matemático que a IA entende os traços implícitos da linguagem humana sabendo sem você explicar, que "Filho" e "Ententeado" estão no campo parental.

#### 📐 A Matemática dos Embeddings e o "Cosseno"
* **Word Embeddings**: Diferente das Strings nativas ou ID's cegos, cada token na Fase 3 se transforma em um "Ponteiro Gráfico Multidimensional" (Array puro em `float64`). 
* **Cosign Similarity (Similaridade do Cosseno)**: Implementamos o sagrado algorítmo por trás dos mecanismos de busca de vetores da atualidade. A matemática avalia a angulação escalar entre dois vetores. Se dois vetores (palavras) fluem na exata mesma direção semântica/contextual da frase, eles têm máxima similaridade (Perto de `1.0`). Se dão em ângulos obtusos ou opostos, variam para nulo (`0.0`) ou extremos paradoxos/antônimos (`-1.0`).

**🤖 Exemplo Prático:**
Simulamos a memória da I.A com as grandezas pré-assimiladas (como *Grau de Riqueza, Pertencimento na Realeza e Nível-Roedor*)... Comparando a angulação geométrica de palavras da base com a nossa fórmula construída do zero, ela prova os parentescos ideológicos com maestria!

```text
============ FASE 3: SIMILARIDADE SEMÂNTICA (VECTORES) ============
Testando familiaridade semântica matemática entre 'Rei' 👑 e 'Rainha' 👸...
>> Grau de Similaridade: 0.996928 (Próximo de 1.0 é extremo parentesco semântico!)

Testando familiaridade semântica matemática entre 'Rei' 👑 e 'Rato' 🐭...
>> Grau de Similaridade: 0.155694 (Próximo ou Menor que 0.0 significa que não compartilham quase nenhum sentido)
```

---

### Fase 4: O Mecanismo de Busca / Retrieval (Concluído)

De posse das nossas matrizes matemáticas no espaço e da nossa fórmula de similaridade (Fase 3), precisávamos apenas da inteligência central interligadora do pacote `search/`. Entra em cena a varredura do modelo `NearestNeighbor` (Busca pelo Vizinho Mais Próximo).

#### 🎯 Como o Search funciona?
* Em arquiteturas empresariais modernas como as famosas **RAGs** (*Retrieval-Augmented Generation*), antes de a IA tentar redigir um parágrafo pra você fingindo que sabe o assunto, ela primeiro varre seus arquivos PDF ou planilhas procurando vetores "próximos" da sua pergunta para usar de embasamento na geração.
* Neste módulo construímos uma busca vetorial de varredura (*Linear Scan*). Nossa função recebe um array multidimensional misterioso de entrada (a pergunta convertida) e joga ela contra toda a memória da IA avaliando quem responde à *Similaridade de Cosseno*.

**🤖 Exemplo Mocado de Retorno Vizinho Mais Próximo:**
Simulamos na classe `main.go` um "vetor de pergunta virtual" sobre um novo indivíduo na realeza de alta riqueza, que seria um Príncipe. Ao pedir para ela comparar com a memória atual (Rei, Rainha e Rato), vemos o algoritmo não fazer deduções gramaticais, mas sim matemáticas espaciais absolutas para eleger o vencedor:

```text
==================================================
========= FASE 4: BUSCA SEMÂNTICA (SEARCH) =======
==================================================
[INFO] Consultando a Base Vectorial: [Rei, Rainha, Rato]
[PERGUNTA DA IA] Qual palavra no meu cérebro matemático melhor se encaixa neste vetor misterioso (Alto Poder Real)?

🤖>> Resposta Final: A palavra que mais faz sentido é a 'Rainha' (Semelhança Cosseno: 99.95%)
==================================================
```
