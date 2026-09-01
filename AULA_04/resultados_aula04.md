# Relatório de Avaliação NLU - SAC Móveis Residenciais

## 1. Tabela Comparativa de Métricas (Dados de Teste)

| Modelo | Acurácia Geral | F1-Score (Weighted) | Principais Erros na Matriz |
| :--- | :--- | :--- | :--- |
| **KNN (K=3)** | 100% | 1.00 | Nenhum erro. Acertou todas as 30 frases de teste. |
| **Decision Tree** | 80% | 0.80 | Errou 6 frases: confundiu logística com vendas, e teve trocas entre reclamações, suporte e trocas. |

## 2. Análise dos Testes de Entrada (`input()`)
- **Comportamento do KNN (10 testes):** Acertou com facilidade as frases normais (100% de confiança). Quando a frase era fora do assunto ou confusa, a confiança caiu e o fallback foi acionado corretamente para mandar ao atendente humano.
- **Comportamento da Decision Tree (8 testes):** Teve mais dificuldade. Por olhar palavras soltas, ela deu 100% de certeza mesmo em frases que classificou errado ou fora do contexto, falhando em acionar o fallback quando deveria.

## 3. Veredito Final
- **Melhor modelo para este projeto:** KNN (K=3)
- **Justificativa técnica:** O KNN teve melhor desempenho geral (100% contra 80% de acurácia) e é mais confiável para entender textos com TF-IDF, além de acionar o fallback de forma muito mais segura.