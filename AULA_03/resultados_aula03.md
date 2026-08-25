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
