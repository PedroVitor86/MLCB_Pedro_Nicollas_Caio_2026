# ==============================================================================
# ATIVIDADE 2: CHATBOT VERSÃO 2 (DECISION TREE)
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Carregar dataset do CSV
df = pd.read_csv('dataset_moveis_100.csv')

# 2. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'], df['intencao'], test_size=0.30, random_state=42, stratify=df['intencao']
)

# 3. Pipeline com TfidfVectorizer e DecisionTreeClassifier
pipeline_dt = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 4. Treinar a pipeline com os dados de treino
pipeline_dt.fit(X_train, y_train)

# 5. Predições e métricas nos dados de teste
y_pred = pipeline_dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

LIMIAR_CONFIANCA = 0.50

print("\n=== INICIANDO BATERIA DE TESTES (8 INPUTS OBRIGATÓRIOS) ===")

for i in range(1, 9):
    print(f"\n[Teste {i}/8]")
    
    frase = input("Digite a frase do cliente: ").strip()
    
    probs = pipeline_dt.predict_proba([frase])[0]
    maior_prob = np.max(probs)
    intencao = pipeline_dt.predict([frase])[0]
    
    if maior_prob >= LIMIAR_CONFIANCA:
        print(f"Intenção: {intencao} (Confiança: {maior_prob:.2f})")
    else:
        print("Desculpe, não entendi sua solicitação. Encaminhando você para um atendente humano...")