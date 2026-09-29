LAB 1 - Regressao Logistica (Chatbot Bancario)
Teste 1: "Quero transferir 300 reais para minha mae" {'status': 'FALLBACK', 'confianca': 0.4191, 'mensagem': 'Nao entendi sua solicitacao. Transferindo para atendente humano.'}

Teste 2: "Qual o saldo da minha conta" {'status': 'FALLBACK', 'confianca': 0.437, 'mensagem': 'Nao entendi sua solicitacao. Transferindo para atendente humano.'}

Teste 3: "Preciso bloquear meu cartao agora" {'status': 'FALLBACK', 'confianca': 0.4915, 'mensagem': 'Nao entendi sua solicitacao. Transferindo para atendente humano.'}

Teste 4: "Qual a temperatura hoje em Curitiba" {'status': 'FALLBACK', 'confianca': 0.3306, 'mensagem': 'Nao entendi sua solicitacao. Transferindo para atendente humano.'}

Testes com threshold menor (0.35):

Teste 5: "Quero transferir 300 reais para minha mae" {'status': 'SUCESSO', 'intencao': 'transferencia', 'confianca': 0.4191, 'valor_extraido': 300.0}

Teste 6: "Qual o saldo da minha conta" {'status': 'SUCESSO', 'intencao': 'consulta_saldo', 'confianca': 0.437, 'valor_extraido': None}

Teste 7: "Preciso bloquear meu cartao agora" {'status': 'SUCESSO', 'intencao': 'bloquear_cartao', 'confianca': 0.4915, 'valor_extraido': None}

Teste 8: "Qual a temperatura hoje em Curitiba" {'status': 'FALLBACK', 'confianca': 0.3306, 'mensagem': 'Nao entendi sua solicitacao. Transferindo para atendente humano.'}

Com threshold 0.55 tudo cai no fallback por causa do dataset pequeno. Abaixando pra 0.35 o modelo acerta as intencoes e extrai o valor de 300 reais no teste 5

LAB 2 - Naive Bayes (Triagem de Chamados)
Teste 1: "Urgente: servidor caiu no protocolo INC-9982" {'status': 'FALLBACK', 'confianca': 0.5456, 'mensagem': 'Nao foi possivel classificar. Encaminhando para analise manual.'}

Teste 2: "Como faco para trocar minha senha" {'status': 'FALLBACK', 'confianca': 0.5583, 'mensagem': 'Nao foi possivel classificar. Encaminhando para analise manual.'}

Teste 3: "Quero saber o horario de funcionamento" {'status': 'FALLBACK', 'confianca': 0.3717, 'mensagem': 'Nao foi possivel classificar. Encaminhando para analise manual.'}

Teste 4: "Sistema fora do ar protocolo INC-4455" {'status': 'FALLBACK', 'confianca': 0.5009, 'mensagem': 'Nao foi possivel classificar. Encaminhando para analise manual.'}

Testes com threshold menor (0.45):

Teste 5: "Urgente: servidor caiu no protocolo INC-9982" {'status': 'SUCESSO', 'intencao': 'incidente_critico', 'confianca': 0.5456, 'protocolo': 'INC-9982'}

Teste 6: "Como faco para trocar minha senha" {'status': 'SUCESSO', 'intencao': 'duvida_suporte', 'confianca': 0.5583, 'protocolo': None}

Teste 7: "Sistema fora do ar protocolo INC-4455" {'status': 'SUCESSO', 'intencao': 'incidente_critico', 'confianca': 0.5009, 'protocolo': 'INC-4455'}

Com threshold 0.60 cai tudo no fallback. Abaixando pra 0.45 o Naive Bayes classifica certo e extrai os protocolos INC-9982 e INC-4455

LAB 3 - Decision Tree (E-Commerce)
Teste 1: "Quero rastrear o pedido BR987654321" {'status': 'SUCESSO', 'intencao': 'rastrear_pedido', 'confianca': 1.0, 'codigo_rastreio': 'BR987654321'}

Teste 2: "Preciso cancelar minha compra" {'status': 'SUCESSO', 'intencao': 'cancelar_compra', 'confianca': 1.0, 'codigo_rastreio': None}

Teste 3: "Qual o horario de atendimento da loja fisica" {'status': 'FALLBACK', 'confianca': 1.0, 'mensagem': 'Nao entendi. Vou transferir para um atendente.'}

Teste 4: "Onde esta minha encomenda BR123456789" {'status': 'SUCESSO', 'intencao': 'rastrear_pedido', 'confianca': 1.0, 'codigo_rastreio': 'BR123456789'}

A Decision Tree deu 100% de confianca em todos os testes. Classificou certo as intencoes, extraiu os codigos de rastreio BR987654321 e BR123456789, e o teste 3 caiu no fallback porque detectou fora_escopo
