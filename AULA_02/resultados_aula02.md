
LAB 01

RESULTADOS:
--- RESULTADOS DO LAB 01 ---
Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]

PERGUNTA 1:
Os resultados foram quase todos corretos, porem a intenção do primeiro era consultar dinheiro e saiu a opção para fazer pix

PERGUNTA 2:
Para melhorar o resultado do algoritmo, seria necessário aumentar a quantidade e a variedade dos exemplos utilizados no treinamento. Também seria importante adicionar diferentes formas de escrever uma mesma intenção, para que o modelo consiga reconhecer melhor mensagens novas.

PERGUNTA 3:
A LogisticRegression é responsável por classificar as mensagens. Ela aprende com os exemplos fornecidos no treinamento e, quando recebe uma nova mensagem, tenta identificar a qual intenção ela pertence.


## LAB 02 - Naive Bayes e Probabilidades

### Resultados da execução

--- RESULTADOS DO LAB 02 ---

Mensagem de Teste: 'Gostaria de devolver o produto que comprei'

Intenção Predita: troca_devolucao

--- Distribuição de Probabilidades por Classe ---

Classe [duvida_frete]: 27.99%

Classe [rastrear_pedido]: 24.54%

Classe [troca_devolucao]: 47.46%

### 1 - Avaliação dos resultados

O resultado foi correto, pois a mensagem "Gostaria de devolver o produto que comprei" está relacionada à devolução de um produto. O algoritmo classificou corretamente a mensagem como "troca_devolucao". A maior probabilidade também foi atribuída a essa classe, com 47,46%.

### 2 - Como melhorar os resultados

Para melhorar os resultados do algoritmo, seria necessário aumentar a quantidade e a variedade dos exemplos utilizados no treinamento. Dessa forma, o modelo poderia aprender diferentes formas de escrever mensagens relacionadas a cada intenção e realizar classificações mais precisas.

### 3 - Função do Naive Bayes

O Naive Bayes é o algoritmo responsável por classificar a mensagem com base em probabilidades. Ele analisa os padrões presentes nos exemplos de treinamento e calcula a probabilidade de uma nova mensagem pertencer a cada classe. Neste caso, a classe "troca_devolucao" apresentou a maior probabilidade, com 47,46%, sendo escolhida pelo algoritmo.

## LAB 03 - Árvore de Decisão

### Código corrigido

# ============================================================
# LAB 03 - AULA 02 (MLCB): Preencha os blocos TODO
# ============================================================
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset de Suporte Técnico
dados_tech = {
    'mensagem': [
        'Esqueci minha senha de acesso', 'Não consigo entrar no sistema', 'Como redefinir minha senha?',
        'A internet esta muito lenta', 'Sem conexao de rede no escritorio', 'Minha conexao caindo toda hora',
        'Impressora nao esta funcionando', 'Nao consigo imprimir documentos', 'Impressora travada com papel'
    ],
    'intencao': [
        'reset_senha', 'reset_senha', 'reset_senha',
        'problema_conexao', 'problema_conexao', 'problema_conexao',
        'suporte_impressora', 'suporte_impressora', 'suporte_impressora'
    ]
}

df3 = pd.DataFrame(dados_tech)

# TODO 1: Separe o dataset em X (coluna 'mensagem') e y (coluna 'intencao')
X = df3['mensagem']
y = df3['intencao']

# TODO 2: Realize a divisão em treino (70%) e teste (30%) com random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# TODO 3: Instancie o CountVectorizer e ajuste/transforme os dados de treino e teste
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# TODO 4: Instancie o DecisionTreeClassifier e treine o modelo com .fit()
modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(X_train_vec, y_train)

# TODO 5: Gere as predições para o X_test_vec e exiba a acurácia
predicoes = modelo_arvore.predict(X_test_vec)

acuracia = accuracy_score(y_test, predicoes)

print(f"Acurácia do Modelo: {acuracia * 100:.2f}%")

#========== PRODUÇÃO DO RELATÓRIO:==============
# Para a entrega completa deste LAB03 você precisa colar o código corrigido com os TODOs preenchidos, a acurácia obtida e responder:
# 1 - Qual foi a acurácia obtida pelo modelo no conjunto de teste e por que, em um dataset tão pequeno (9 exemplos), essa métrica pode ser enganosa?
# 2 - Como o modelo de Árvore de Decisão (DecisionTreeClassifier) toma a decisão de separar as intenções do usuário?
# 3 - Qual é o risco de utilizar uma Árvore de Decisão sem limite de profundidade (max_depth) em datasets de texto maiores?

# Todos os resultados devem ser inseridos no arquivo resultados_aula02.md

#========== FIM ==============

### Resultado

Acurácia do Modelo: 33.33%

### 1 - Avaliação da acurácia

A acurácia obtida pelo modelo foi de 33,33%. Essa métrica pode ser enganosa porque o dataset possui apenas 9 exemplos, fazendo com que o conjunto de teste tenha poucos dados. Dessa forma, um único acerto ou erro pode causar uma grande alteração na porcentagem da acurácia. Portanto, seria necessário utilizar um conjunto de dados maior para obter uma avaliação mais confiável.

### 2 - Funcionamento da Árvore de Decisão

O DecisionTreeClassifier toma decisões criando divisões com base nas características presentes nos dados. No caso deste laboratório, as palavras transformadas pelo CountVectorizer são utilizadas para encontrar padrões que ajudam a separar as diferentes intenções, como reset_senha, problema_conexao e suporte_impressora.

### 3 - Risco de não limitar a profundidade

O principal risco é o overfitting, no qual a árvore fica muito complexa e passa a memorizar os exemplos do conjunto de treinamento em vez de aprender padrões que possam ser aplicados a novos dados. Em datasets de texto maiores, isso pode prejudicar a capacidade de generalização do modelo. Definir um limite para max_depth pode ajudar a controlar a complexidade da árvore.

# LAB 04

## RESULTADOS:

--- RESULTADOS DO LAB 04 ---

Quantidade de frases: 15

Distribuição das intenções:

comprar_passagem    5
cancelar_reserva    5
falar_atendente     5

Quantidade de exemplos para treinamento: 10
Quantidade de exemplos para teste: 5

Vetorização realizada com sucesso!

Modelo treinado com sucesso!

RESULTADO DO TESTE
==================
Acurácia: [COLOCAR A ACURÁCIA QUE APARECEU NO COLAB]

Relatório de classificação:
[COLOCAR O RELATÓRIO QUE APARECEU NO COLAB]


--- PREDIÇÕES PARA FRASES INÉDITAS ---

Frase: 'Gostaria de adquirir um bilhete de avião'
Intenção prevista: [COLOCAR O RESULTADO DO COLAB]

Frase: 'Não quero mais minha reserva'
Intenção prevista: [COLOCAR O RESULTADO DO COLAB]

Frase: 'Quero conversar com uma pessoa da agência'
Intenção prevista: [COLOCAR O RESULTADO DO COLAB]


### 1 - Avaliação dos resultados

O modelo conseguiu realizar a classificação das frases de teste e também fez previsões para frases inéditas. Os resultados mostram que o algoritmo conseguiu identificar padrões relacionados às intenções presentes no dataset. As previsões das frases inéditas foram utilizadas para verificar se o modelo conseguia reconhecer diferentes formas de escrita das mesmas intenções.

### 2 - Como melhorar os resultados

Para melhorar os resultados do algoritmo, seria necessário aumentar a quantidade e a variedade das frases utilizadas no treinamento. Também seria importante adicionar diferentes formas de escrever uma mesma intenção, incluindo frases mais curtas, mais longas e com palavras diferentes. Dessa forma, o modelo poderia aprender mais padrões e realizar classificações mais precisas.

### 3 - Função do TfidfVectorizer

O TfidfVectorizer é responsável por transformar as frases em valores numéricos que podem ser utilizados pelo algoritmo de classificação. Ele analisa as palavras presentes nos textos e atribui valores de acordo com a importância delas dentro do conjunto de dados.

### 4 - Função da LogisticRegression

A LogisticRegression é responsável por classificar as mensagens de acordo com as intenções aprendidas durante o treinamento. Depois de receber os textos transformados pelo TfidfVectorizer, o algoritmo identifica padrões e utiliza esses padrões para prever a intenção de novas mensagens.

