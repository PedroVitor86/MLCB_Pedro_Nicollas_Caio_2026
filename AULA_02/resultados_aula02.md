
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

