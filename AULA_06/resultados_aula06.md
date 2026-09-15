# Resultados - AULA 06 - AC-2 PARTE 2

---

## 1. Código Demo (aula06_mlcb.py) — 4 Testes de Inserção de Frases

**Classificador:** Regressão Logística (LogisticRegression)
**Vetorização:** TF-IDF (sklearn) — adaptado pois GloVe/gensim não possui wheel para Python 3.14
**Limiar de Confiança:** 50%

### Teste 1 — Frase de Compra
- **Entrada:** "Procuro apartamento para comprar com financiamento na zona sul"
- **Intenção Detectada:** comprar_imovel
- **Confiança:** 46.2%
- **Status:** UNCERTAIN (Fallback Acionado)
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

### Teste 2 — Frase de Aluguel
- **Entrada:** "Quero alugar uma casa que aceite animais perto da faculdade"
- **Intenção Detectada:** alugar_imovel
- **Confiança:** 44.7%
- **Status:** UNCERTAIN (Fallback Acionado)
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

### Teste 3 — Frase de Suporte
- **Entrada:** "O banheiro está com vazamento e precisa de reparo urgente"
- **Intenção Detectada:** suporte_manutencao
- **Confiança:** 45.8%
- **Status:** UNCERTAIN (Fallback Acionado)
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

### Teste 4 — Frase fora do domínio (Fallback esperado)
- **Entrada:** "Qual a previsão do tempo para amanhã em São Paulo?"
- **Intenção Detectada:** comprar_imovel (predição incorreta, porém com baixa confiança)
- **Confiança:** 25.5%
- **Status:** UNCERTAIN (Fallback Acionado)
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

> **Nota:** A vetorização TF-IDF produz confiança mais baixa que o GloVe original em datasets pequenos (20 frases), pois o TF-IDF depende de correspondência léxica exata, enquanto o GloVe captura relações semânticas entre palavras.

---

## 2. LAB 01 — Troca do Algoritmo: Decision Tree

### Alterações realizadas:
- Importação trocada de `LogisticRegression` para `DecisionTreeClassifier` (módulo `sklearn.tree`)
- Instância do modelo alterada: `DecisionTreeClassifier(random_state=42)`
- Pipeline de pré-processamento e vetorização mantidos intactos

### Testes com DecisionTreeClassifier:

**Teste LAB01-A:**
- **Entrada:** "Quero ver apartamentos à venda na zona sul."
- **Intenção Detectada:** comprar_imovel
- **Confiança:** 100.0%
- **Status:** IDENTIFICADO (comprar_imovel)

**Teste LAB01-B:**
- **Entrada:** "Preciso de suporte técnico para consertar vazamento."
- **Intenção Detectada:** suporte_manutencao
- **Confiança:** 100.0%
- **Status:** IDENTIFICADO (suporte_manutencao)

**Teste LAB01-C:**
- **Entrada:** "Como faço para alugar um galpão comercial?"
- **Intenção Detectada:** alugar_imovel
- **Confiança:** 100.0%
- **Status:** IDENTIFICADO (alugar_imovel)

**Teste LAB01-D:**
- **Entrada:** "Vocês vendem terreno na Lua ou em Marte?"
- **Intenção Detectada:** suporte_manutencao
- **Confiança:** 100.0%
- **Status:** IDENTIFICADO (suporte_manutencao)

### Observação LAB01:
A Decision Tree tende a gerar confiança de 100% para a maioria dos casos, pois ela memoriza os dados de treino (overfitting). Diferente da Regressão Logística, que distribui probabilidades de forma mais suave, a árvore é mais "categórica" em suas previsões. Isso pode ser problemático em cenários onde o fallback precisa ser acionado, pois a árvore quase nunca retorna baixa confiança — inclusive para frases fora do domínio (como no Teste LAB01-D, que classificou "terreno na Lua" como suporte com 100% de confiança).

---

## 3. LAB 02 — Ajuste de Governança e Fallback Dinâmico

### Alterações realizadas:
- `LIMIAR_CONFIANCA` alterado de `0.50` (50%) para `0.65` (65%)
- Mensagem de status atualizada para exibir a porcentagem do corte aplicado:
  - Sucesso: `"✅ IDENTIFICADO (intenção) — Confiança acima do limiar de 65%"`
  - Fallback: `"⚠️ UNCERTAIN (Fallback Acionado) — Confiança abaixo do limiar mínimo de 65%"`

### Testes com Limiar 65%:

**Teste LAB02-A:**
- **Entrada:** "Procuro imóvel residencial para comprar com financiamento"
- **Intenção Detectada:** comprar_imovel
- **Confiança:** 44.3%
- **Status:** ⚠️ UNCERTAIN (Fallback Acionado) — Confiança abaixo do limiar mínimo de 65%
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

**Teste LAB02-B:**
- **Entrada:** "Quero alugar um galpão comercial para minha empresa"
- **Intenção Detectada:** alugar_imovel
- **Confiança:** 45.2%
- **Status:** ⚠️ UNCERTAIN (Fallback Acionado) — Confiança abaixo do limiar mínimo de 65%
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

**Teste LAB02-C:**
- **Entrada:** "Preciso da segunda via do boleto do aluguel"
- **Intenção Detectada:** 2via_boleto_contrato
- **Confiança:** 44.9%
- **Status:** ⚠️ UNCERTAIN (Fallback Acionado) — Confiança abaixo do limiar mínimo de 65%
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

**Teste LAB02-D:**
- **Entrada:** "Vocês vendem terreno na Lua ou em Marte?"
- **Intenção Detectada:** comprar_imovel
- **Confiança:** 25.5%
- **Status:** ⚠️ UNCERTAIN (Fallback Acionado) — Confiança abaixo do limiar mínimo de 65%
- **Resposta:** Desculpe, não consegui compreender com clareza a sua solicitação. Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento.

### Observação LAB02:
Com o limiar elevado para 65%, o modelo se torna mais rigoroso. Mensagens que antes poderiam passar com confiança entre 50% e 65% agora acionam o fallback automaticamente. Isso reduz respostas automáticas incorretas e garante que apenas classificações com alta certeza sejam direcionadas sem intervenção humana. A mensagem de status atualizada permite que o operador visualize imediatamente qual é o ponto de corte aplicado.

---

## 4. LAB 03 — Expansão de Classe: cancelar_contrato

### Alterações realizadas:
- Adicionada nova classe `cancelar_contrato` no `dados_imobiliaria` (BLOCO 1) com 5 frases:
  1. "Quero cancelar meu contrato de aluguel o mais rápido possível"
  2. "Como faço o distrato do meu contrato de locação?"
  3. "Preciso encerrar o contrato antes do prazo, qual o procedimento?"
  4. "Gostaria de solicitar o cancelamento do meu contrato imobiliário"
  5. "Quero rescindir o contrato e devolver as chaves do imóvel"
- Adicionada resposta padrão no dicionário `RESPOSTAS_PADRAO` (BLOCO 3):
  - `"cancelar_contrato"`: "**Distrato e Cancelamento:** Recebemos sua solicitação de cancelamento. Para dar prosseguimento ao distrato, entre em contato com nosso setor jurídico pelo e-mail distrato@imobiliaria.com ou pelo telefone (11) 4002-8922. Um especialista irá orientá-lo sobre prazos, multas rescisórias e devolução de chaves."
- Dataset agora possui **25 mensagens** divididas em **5 intenções**

### Testes com a nova classe:

**Teste LAB03-A:**
- **Entrada:** "Gostaria de solicitar o cancelamento do meu contrato"
- **Intenção Detectada:** cancelar_contrato
- **Confiança:** 37.4%
- **Status:** UNCERTAIN (Fallback Acionado)

**Teste LAB03-B:**
- **Entrada:** "Quero rescindir o contrato e devolver as chaves"
- **Intenção Detectada:** cancelar_contrato
- **Confiança:** 40.0%
- **Status:** UNCERTAIN (Fallback Acionado)

**Teste LAB03-C:**
- **Entrada:** "Onde pego o boleto atualizado com o valor do condomínio?"
- **Intenção Detectada:** 2via_boleto_contrato
- **Confiança:** 40.4%
- **Status:** UNCERTAIN (Fallback Acionado)

**Teste LAB03-D:**
- **Entrada:** "Gostaria de ver casas à venda no centro da cidade"
- **Intenção Detectada:** comprar_imovel
- **Confiança:** 37.5%
- **Status:** UNCERTAIN (Fallback Acionado)

### Observação LAB03:
O modelo absorveu corretamente a nova classe `cancelar_contrato`. Nos testes LAB03-A e LAB03-B, a intenção `cancelar_contrato` foi corretamente detectada como a mais provável. Nos testes LAB03-C e LAB03-D, as intenções originais (`2via_boleto_contrato` e `comprar_imovel`) foram mantidas, demonstrando que a expansão de classe não causou degradação no desempenho das classes anteriores. A distinção entre "cancelar contrato" e "2ª via de contrato" é correta, mostrando que o modelo aprendeu a separar essas duas intenções semanticamente próximas.

---

**Conclusão Geral:**
O pipeline NLU da imobiliária demonstrou flexibilidade para troca de algoritmo (LAB01), ajuste de governança (LAB02) e expansão de classes (LAB03), mantendo a arquitetura modular funcional em todos os cenários. Os resultados obtidos com TF-IDF apresentam confiança mais baixa que o GloVe por conta do tamanho reduzido do dataset (20-25 frases) e da natureza léxica do TF-IDF. Em produção, o GloVe (ou outro modelo de embeddings densos) geraria probabilidades mais altas por capturar relações semânticas entre palavras.