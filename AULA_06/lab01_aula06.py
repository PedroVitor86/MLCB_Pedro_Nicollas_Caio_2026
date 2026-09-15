# LAB 01 - Troca do Algoritmo de Classificação: Decision Tree
# Substituição da Regressão Logística por Árvore de Decisão (DecisionTreeClassifier)

# Instalação (executar apenas 1x no terminal):
# pip install gensim scikit-learn spacy pandas numpy gradio
# python -m spacy download pt_core_news_sm

# ===== BLOCO 1: Preparação do Ambiente e Carga do Dataset =====

import re
import numpy as np
import pandas as pd
import spacy
import gensim.downloader as api
from sklearn.tree import DecisionTreeClassifier  # <<< ALTERAÇÃO LAB01: trocado LogisticRegression por DecisionTreeClassifier
import gradio as gr

# 1. Carregamento do modelo de linguagem em português (Spacy)
print("Carregando modelo morfológico Spacy (pt_core_news_sm)")
nlp = spacy.load("pt_core_news_sm")

# 2. Carregamento dos Word Embeddings (Gensim / GloVe de 50 dimensões)
print("Carregando espaço vetorial denso de embeddings modelo Glove")
word_vectors = api.load("glove-wiki-gigaword-50")

# 3. Base de Dados Supervisionada: SAC de Imóveis
dados_imobiliaria = [
    # Intenção: comprar_imovel
    ("Quero comprar um apartamento de 3 quartos com varanda", "comprar_imovel"),
    ("Gostaria de ver casas à venda no centro da cidade", "comprar_imovel"),
    ("Qual o preço médio para compra de cobertura com piscina?", "comprar_imovel"),
    ("Procuro imóvel residencial para comprar com financiamento", "comprar_imovel"),
    ("Vocês têm sobrados à venda na zona sul?", "comprar_imovel"),

    # Intenção: alugar_imovel
    ("Procurando kitnet para alugar perto da faculdade", "alugar_imovel"),
    ("Qual o valor do aluguel deste apartamento de 2 dormitórios?", "alugar_imovel"),
    ("Quero alugar um galpão comercial para minha empresa", "alugar_imovel"),
    ("Quais imóveis estão disponíveis para locação imediata?", "alugar_imovel"),
    ("Preciso de uma casa para alugar que aceite animais", "alugar_imovel"),

    # Intenção: suporte_manutencao
    ("O chuveiro do apartamento alugado queimou, como pedir conserto?", "suporte_manutencao"),
    ("Muro da casa está com infiltração e vazamento de água", "suporte_manutencao"),
    ("Preciso do contato do encanador para reparo na cozinha", "suporte_manutencao"),
    ("A porta da varanda quebrou, quem faz a manutenção?", "suporte_manutencao"),
    ("Vazamento no teto do banheiro precisa de reparo urgente", "suporte_manutencao"),

    # Intenção: 2via_boleto_contrato
    ("Como faço para baixar a segunda via do boleto do aluguel?", "2via_boleto_contrato"),
    ("Não recebi o boleto deste mês para pagamento", "2via_boleto_contrato"),
    ("Preciso do informe de rendimentos e cópia do contrato", "2via_boleto_contrato"),
    ("Onde pego o boleto atualizado com o valor do condomínio?", "2via_boleto_contrato"),
    ("Quero solicitar a segunda via do recibo de pagamento", "2via_boleto_contrato")
]

df = pd.DataFrame(dados_imobiliaria, columns=["mensagem", "intencao"])
print(f"Dataset carregado com {len(df)} mensagens divididas em {df['intencao'].nunique()} intenções.")


# ===== BLOCO 2: Esteira NLU (Pré-processamento e Vectorization) =====

def preprocessar_texto(texto: str) -> str:
    texto_limpo = texto.lower()
    texto_limpo = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', texto_limpo)
    doc = nlp(texto_limpo)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_space and len(token.text) > 1
    ]
    return " ".join(tokens)

def extrair_sentence_embedding(texto_limpo: str, modelo_emb) -> np.ndarray:
    palavras = texto_limpo.split()
    vetores = [modelo_emb[p] for p in palavras if p in modelo_emb]
    if len(vetores) == 0:
        return np.zeros(modelo_emb.vector_size)
    return np.mean(vetores, axis=0)

# Aplicação no Dataset
df['mensagem_limpa'] = df['mensagem'].apply(preprocessar_texto)

X_densos = np.array([
    extrair_sentence_embedding(txt, word_vectors) for txt in df['mensagem_limpa']
])

y = df['intencao'].values


# ===== BLOCO 3: Treinamento do Modelo e Mapeamento de Respostas Padrão =====

# <<< ALTERAÇÃO LAB01: DecisionTreeClassifier no lugar de LogisticRegression >>>
modelo_nlu = DecisionTreeClassifier(random_state=42)
modelo_nlu.fit(X_densos, y)

print("Modelo supervisionado treinado com DecisionTreeClassifier!")

# Base de Conhecimento: Respostas Padrão de Negócio do SAC
RESPOSTAS_PADRAO = {
    "comprar_imovel": (
        "**Atendimento de Vendas:** Ficamos felizes com seu interesse! "
        "Você pode conferir nosso catálogo de imóveis à venda em nosso site www.imobiliaria.com/vendas "
        "ou aguardar que um de nossos corretores entrará em contato em instantes."
    ),
    "alugar_imovel": (
        "**Atendimento de Locação:** Temos ótimas opções disponíveis! "
        "Acesse www.imobiliaria.com/aluguel para filtrar por região e valor. "
        "Para agendar uma visita, envie o código do imóvel por aqui."
    ),
    "suporte_manutencao": (
        "**Suporte e Manutenção:** Sentimos muito pelo inconveniente. "
        "Por favor, abra um chamado urgente em nosso portal do inquilino (www.imobiliaria.com/manutencao) "
        "anexando fotos ou vídeos do problema para acionarmos nossos prestadores."
    ),
    "2via_boleto_contrato": (
        "**Financeiro e Contratos:** Para acessar boletos ou documentos, "
        "acesse a Área do Cliente em www.imobiliaria.com/cliente informando seu CPF e senha. "
        "Lá você baixa a 2ª via atualizada em segundos."
    )
}


# ===== BLOCO 4: Motor de Inferência com Lógica de Fallback =====

LIMIAR_CONFIANCA = 0.50

def processar_atendimento_sac(mensagem_usuario: str):
    if not mensagem_usuario or not mensagem_usuario.strip():
        return "N/A", "0.0%", " Aguardando mensagem...", "Aguardando entrada do usuário..."

    # Step 1: Pré-processamento
    msg_limpa = preprocessar_texto(mensagem_usuario)

    # Step 2: Vetorização
    vetor_input = extrair_sentence_embedding(msg_limpa, word_vectors).reshape(1, -1)

    # Step 3: Predição de Probabilidades
    probabilidades = modelo_nlu.predict_proba(vetor_input)[0]
    idx_maior_prob = np.argmax(probabilidades)
    confianca = probabilidades[idx_maior_prob]
    intencao_detectada = modelo_nlu.classes_[idx_maior_prob]

    percentual_confianca = f"{confianca * 100:.1f}%"

    # Step 4: Regra de Decisão / Fallback
    if confianca >= LIMIAR_CONFIANCA:
        classificacao_status = f" IDENTIFICADO ({intencao_detectada})"
        texto_resposta = RESPOSTAS_PADRAO[intencao_detectada]
    else:
        classificacao_status = " UNCERTAIN (Fallback Acionado)"
        texto_resposta = (
            "Desculpe, não consegui compreender com clareza a sua solicitação. "
            "Estou transferindo agora mesmo sua conversa para um de nossos atendentes. Por favor, aguarde um momento."
        )

    card_resposta = f"""
    <div style="background-color: #f0f4f9; border-left: 5px solid #2b5c8f; padding: 15px; border-radius: 8px; margin-top: 10px;">
        <h4 style="margin: 0 0 8px 0; color: #2b5c8f;"> Resposta Automática do SAC:</h4>
        <p style="margin: 0; font-size: 15px; color: #1a1a1a;">{texto_resposta}</p>
    </div>
    """

    return intencao_detectada, percentual_confianca, classificacao_status, card_resposta


# ===== BLOCO 5: Construção da Interface Gráfica Interativa (Gradio) =====

with gr.Blocks(theme=gr.themes.Soft(), title="SAC Imobiliário - LAB01 Decision Tree") as app:
    gr.Markdown(
        """
        #  SAC Imobiliário — ChatBot Inteligente (LAB01 - Decision Tree)
        *Será um prazer atendê-lo. Digite a seguir a sua necessidade*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  Mensagem do Cliente")
            input_texto = gr.Textbox(
                lines=4,
                placeholder="Ex: Preciso da segunda via do boleto de aluguel...",
                label="Digite sua necessidade"
            )
            btn_processar = gr.Button(" Processar Mensagem", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    ["Preciso de suporte técnico para consertar vazamento."],
                    ["Quero ver apartamentos à venda na zona sul."],
                    ["Como faço para alugar um galpão comercial?"],
                    ["Gostaria de baixar o boleto do condomínio."],
                    ["Vocês vendem terreno na Lua ou em Marte?"]
                ],
                inputs=input_texto
            )

        with gr.Column(scale=1):
            gr.Markdown("### Painel de Diagnóstico")
            with gr.Row():
                out_intencao = gr.Textbox(label="Intenção", scale=2, interactive=False)
                out_confianca = gr.Textbox(label="Confiança", scale=1, interactive=False)

            out_status = gr.Textbox(label="Status da Decisão", interactive=False)
            out_resposta = gr.HTML(
                value="<div style='padding: 15px; color: #888;'>Aguardando envio de mensagem...</div>",
                label="Resposta da Imobiliária"
            )

    btn_processar.click(
        fn=processar_atendimento_sac,
        inputs=[input_texto],
        outputs=[out_intencao, out_confianca, out_status, out_resposta]
    )

app.launch(debug=True, share=True)
# FIM DO CÓDIGO LAB01
