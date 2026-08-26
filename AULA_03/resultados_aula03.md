--- RESULTADOS DO LAB 01 (AULA 03) ---
Mensagem: 'Preciso urgente da segunda via da fatura'
Intenção Predita: [segunda_via]
Vocabulário Filtrado (sem stopwords): ['2a', '2a via', 'aberto', 'acordo', 'acordo pagar', 'alterar', 'alterar endereço', 'app', 'atrasada', 'atualizo', 'atualizo dados', 'boleto', 'cadastramento', 'dados', 'dados residenciais', 'débito', 'débito aberto', 'dívida', 'emitir', 'emitir segunda', 'endereço', 'endereço cadastramento', 'fatura', 'fatura atrasada', 'fazer', 'fazer um', 'gostaria', 'gostaria alterar', 'negociar', 'negociar pagamento', 'no', 'no app', 'onde', 'onde atualizo', 'pagamento', 'pagamento dívida', 'pagar', 'pagar débito', 'posso', 'posso emitir', 'residenciais', 'residenciais no', 'segunda', 'segunda via', 'um', 'um acordo', 'via', 'via boleto', 'via fatura']
















--- RESULTADOS DO LAB 02 (AULA 03) ---

--- Relatório de Classificação ---
                     precision    recall  f1-score   support

horario_atendimento       0.50      1.00      0.67         1
        localizacao       0.00      0.00      0.00         1
    troca_devolucao       0.00      0.00      0.00         1

           accuracy                           0.33         3
          macro avg       0.17      0.33      0.22         3
       weighted avg       0.17      0.33      0.22         3

--- Matriz de Confusão ---
[[1 0 0]
 [1 0 0]
 [0 1 0]]
                                  RESPOSTAS LAB 02 AULA 03
1. Precision mostra quantas vezes o modelo acertou uma classificação. Recall mostra quantos exemplos corretos daquela classe o modelo conseguiu identificar. O F1-Score é uma média que junta as duas métricas.

2. A diagonal principal mostra os acertos do modelo. Os valores fora dela mostram os erros de classificação.

3. Porque o modelo pode acertar muito a classe que tem mais exemplos e ainda assim errar bastante as outras. Por isso, não é bom analisar apenas a acurácia.


                                        RESPOSTAS LAB 03 AULA 03
   # ============================================================
# LAB 03 - AULA 03 (MLCB): Scikit-Learn Pipeline (Modo TODO)
# ============================================================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados_rh = {
    'mensagem': [
        'Como solicitar minhas ferias?', 'Quero agendar meu periodo de ferias',
        'Onde baixo meu holerite do mes?', 'Preciso do comprovante de rendimentos',
        'Como cadastrar meu atestado medico?', 'Onde envio o atestado de consulta?'
    ],
    'intencao': [
        'solicitar_ferias', 'solicitar_ferias',
        'obter_holerite', 'obter_holerite',
        'enviar_atestado', 'enviar_atestado'
    ]
}

df3 = pd.DataFrame(dados_rh)

# TODO 1: Separe o dataset em X ('mensagem') e y ('intencao')
X = df3['mensagem']
y = df3['intencao']

# TODO 2: Realize o train_test_split com test_size=0.33 e random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# TODO 3: Monte o Pipeline encapsulando o TfidfVectorizer e a LogisticRegression
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=['de', 'o', 'meu', 'minhas'])),
    ('classifier', LogisticRegression())
])


# TODO 4: Treine o pipeline completo com .fit() usando os dados de treino brutos
pipeline.fit(X_train, y_train)


# TODO 5: Faca a predicao nos dados de teste brutos e exiba a acuracia
predicoes = pipeline.predict(X_test)
print(f"Acuracia via Pipeline: {accuracy_score(y_test, predicoes) * 100:.2f}%")


RESULTADO
Acuracia via Pipeline: 0.00%

QUESTÕES
1. O código foi corrigido utilizando o Pipeline para juntar o TfidfVectorizer e a LogisticRegression. A acurácia obtida foi de 0.00%.

2. A principal vantagem do Pipeline é juntar o pré-processamento e o modelo em um único processo, deixando o código mais organizado e simples.

3. Porque o Pipeline aplica automaticamente o mesmo pré-processamento nos dados de treino e de teste, evitando diferenças ou erros entre os dois.


