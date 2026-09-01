# ==============================================================================
# ATIVIDADE 1: CHATBOT VERSÃO 1 (KNN)
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Carregar dataset do CSV
df = pd.read_csv('dataset_moveis_100.csv')

# 2. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'], df['intencao'], test_size=0.30, random_state=42, stratify=df['intencao']
)

# TODO 1: Monte a Pipeline utilizando TfidfVectorizer e KNeighborsClassifier(n_neighbors=3, metric='cosine')
pipeline_knn = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', KNeighborsClassifier(n_neighbors=3, metric='cosine'))
])

# TODO 2: Treine a pipeline com os dados de treino (X_train, y_train)
pipeline_knn.fit(X_train, y_train)

# TODO 3: Gere as predicoes nos dados de teste e exiba o classification_report e a confusion_matrix
y_pred = pipeline_knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

LIMIAR_CONFIANCA = 0.50

print("\n=== INICIANDO BATERIA DE TESTES (10 INPUTS OBRIGATÓRIOS) ===")

for i in range(1, 11):
    print(f"\n[Teste {i}/10]")
    
    # TODO 4: Solicite a frase do usuario via teclado
    frase = input("Digite a frase do cliente: ").strip()
    
    # TODO 5: Extraia as probabilidades e a classe prevista usando predict_proba e predict
    probs = pipeline_knn.predict_proba([frase])[0]
    maior_prob = np.max(probs)
    intencao = pipeline_knn.predict([frase])[0]
    
    # TODO 6: Aplique a regra de decisao:
    # Se maior_prob >= LIMIAR_CONFIANCA: imprima a intencao e a probabilidade.
    # Senao: imprima o Fallback encaminhando para atendimento humano.
    if maior_prob >= LIMIAR_CONFIANCA:
        print(f"Intenção: {intencao} (Confiança: {maior_prob:.2f})")
    else:
        print("Desculpe, não entendi sua solicitação. Encaminhando você para um atendente humano...")