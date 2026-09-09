Relatório de Resultados — Motor NLU (AC-2)1. Teste de Limpeza de Texto (Exercício 1)Frase original: "MEU sofá!!! chegou quebrado e quero DEVOLVER!!!"Frase limpa: "sofá chegar quebrar querer devolver"Resumo: O sistema removeu pontuações, letras maiúsculas e palavras de ligação (meu, e). Sobraram as 5 palavras principais transformadas em suas formas base.2. Testes de Mensagens no Chatbot (Exercício 3)Com o limite de 50% de certeza configurado para o robô responder sozinho:Todas as 5 frases de teste acionaram o FALLBACK_HUMANO com cerca de 25% de confiança.Por que isso aconteceu? O modelo foi treinado com poucas frases (80 exemplos), então ele dividiu a certeza igualmente entre as 4 categorias (25% para cada). Como não atingiu os 50% necessários, o sistema agiu com segurança e encaminhou o cliente para um atendente.3. Comparação dos Modelos (Exercício 4)ModeloAcuráciaPrecisãoF1-ScoreRegressão Logística78,12%80,64%78,39%KNN75,00%75,40%74,90%




4. Respostas das Questões Técnicas
1. Qual modelo foi o melhor?
A Regressão Logística, acertando 78,12% das mensagens de teste contra 75,00% do KNN.

2. Por que os resultados mudam se os dados são os mesmos?
Porque cada modelo "pensa" de um jeito. A Regressão Logística tenta entender o padrão geral do texto, enquanto o KNN só olha para as 3 frases mais parecidas que estão do lado no gráfico.

3. Por que o vetor é tão importante no KNN?
O KNN decide tudo medindo a distância entre as palavras. Se a transformação em vetor não for muito boa, mensagens com assuntos diferentes ficam "grudadas", e o KNN se confunde.

4. Usaria o KNN para um sistema com 100 mil mensagens?
Não. O KNN não aprende uma regra fixa. Toda vez que alguém mandar uma mensagem, ele vai precisar comparar essa frase com as 100 mil cadastradas, uma por uma. Isso deixaria o chatbot travando e muito lento.

5. Qual modelo colocar em produção?
A Regressão Logística. Ela é mais rápida, mais precisa e mede o nível de certeza de forma confiável para chamar um humano quando ficar em dúvida.
