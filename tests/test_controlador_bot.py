"""Testes de integração para o ControladorBot, modularidade White-Label, /metricas e Telegram HTML."""

import json
import tempfile
from pathlib import Path

from src.adaptador_telegram import dividir_mensagem_longa, formatar_para_html_telegram
from src.controlador_bot import ControladorBot
from src.motor_ml import MotorML
from src.registrador_privacidade import RegistradorPrivacidade


def test_comando_start():
    controlador = ControladorBot()
    resposta = controlador.processar_mensagem(1001, "/start")
    assert "CETAM" in resposta.texto
    assert "Agnaldo Marcelo Pereira Fournier" in resposta.texto
    assert "LGPD" in resposta.texto


def test_alternancia_debug_e_resposta():
    with tempfile.TemporaryDirectory() as tmpdir:
        registrador = RegistradorPrivacidade(
            caminho_jsonl=Path(tmpdir) / "log.jsonl",
            caminho_csv=Path(tmpdir) / "log.csv",
        )
        controlador = ControladorBot(registrador_privacidade=registrador)
        id_usuario = 2002

        # Liga debug
        resp_ligar = controlador.processar_mensagem(id_usuario, "/debug")
        assert "ATIVADO" in resp_ligar.texto
        assert controlador.debug_esta_ativo(id_usuario)

        # Pergunta normal com debug ativo
        resp_pergunta = controlador.processar_mensagem(id_usuario, "O que é NLP?")
        assert "MODO DEBUG — RAIO-X DO PIPELINE DE PLN" in resp_pergunta.texto
        assert "conceito_pln_historico" in resp_pergunta.texto
        assert "Resposta do Tutor" in resp_pergunta.texto

        # Desliga debug
        resp_desligar = controlador.processar_mensagem(id_usuario, "/debug")
        assert "DESATIVADO" in resp_desligar.texto
        assert not controlador.debug_esta_ativo(id_usuario)


def test_controle_acesso_regenerar():
    controlador = ControladorBot()
    # Usuário sem whitelist no Telegram
    resp_nao_aut = controlador.processar_mensagem(9999, "/regenerar", canal="telegram")
    assert resp_nao_aut.tipo_acao == "nao_autorizado"
    assert "Acesso Restrito" in resp_nao_aut.texto

    # No canal CLI local, o acesso é liberado para o operador
    resp_cli = controlador.processar_mensagem("console", "/regenerar", canal="cli")
    assert resp_cli.tipo_acao in ("fluxo_regenerar", "texto")


def test_modularidade_white_label_outro_assunto_e_metricas_com_1_amostra():
    """Garante que um aluno pode criar um bot com outro nome/assunto apenas trocando o JSON,
    e que /metricas não quebra ao criar uma intenção nova com apenas 1 amostra."""
    dataset_pizzaria = {
        "metadata": {
            "nome_bot": "PizzaBot Artesanal",
            "nome_curto": "Pizzaiolo",
            "dominio_assunto": "cardápio de pizzas, sabores e entregas",
            "instituicao": "Pizzaria Escola Manaus",
            "exemplos_perguntas": [
                "Quais sabores de pizza vocês têm?",
                "Qual o valor da taxa de entrega?"
            ],
            "respostas_fallback": [
                "Desculpe, só sei responder sobre nosso cardápio de pizzas e entregas!"
            ]
        },
        "intents": [
            {
                "tag": "cardapio_sabores",
                "categoria": "atendimento",
                "descricao": "Sabores de pizza",
                "patterns": [
                    "Quais sabores de pizza vocês têm?",
                    "Me mostra o cardápio de pizzas",
                    "Tem pizza de calabresa e marguerita?"
                ],
                "responses": ["Temos Calabresa, Marguerita, Quatro Queijos e Tucumã!"]
            },
            {
                "tag": "taxa_entrega",
                "categoria": "atendimento",
                "descricao": "Valores e prazo de entrega",
                "patterns": [
                    "Qual o valor da taxa de entrega?",
                    "Vocês entregam em domicílio?",
                    "Quanto custa o frete da pizza?"
                ],
                "responses": ["Nossa taxa de entrega é fixa de R$ 8,00 para toda Manaus!"]
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_json = Path(tmpdir) / "pizzaria.json"
        caminho_json.write_text(json.dumps(dataset_pizzaria, ensure_ascii=False), encoding="utf-8")

        registrador = RegistradorPrivacidade(
            caminho_jsonl=Path(tmpdir) / "log.jsonl",
            caminho_csv=Path(tmpdir) / "log.csv",
        )
        motor = MotorML(caminho_intencoes=caminho_json)
        controlador = ControladorBot(motor_ml=motor, registrador_privacidade=registrador)

        # 1. Verifica se /start e /ajuda assumiram a identidade do PizzaBot
        resp_start = controlador.processar_mensagem(1, "/start")
        assert "PizzaBot Artesanal" in resp_start.texto
        assert "cardápio de pizzas, sabores e entregas" in resp_start.texto

        resp_ajuda = controlador.processar_mensagem(1, "/ajuda")
        assert "PIZZABOT ARTESANAL" in resp_ajuda.texto
        assert "Quais sabores de pizza vocês têm?" in resp_ajuda.texto

        # 2. Cria uma nova intenção via AutoAperfeicoamento com apenas 1 frase de exemplo
        sucesso, _ = controlador.aperfeicoador.criar_nova_intencao(
            tag="horario_funcionamento",
            categoria="atendimento",
            descricao="Horário de abertura",
            primeiro_padrao="Que horas a pizzaria abre hoje?",
            primeira_resposta="Abrimos todos os dias das 18h às 23h30!"
        )
        assert sucesso

        # 3. Verifica se /metricas funciona sem quebrar no StratifiedKFold mesmo tendo classe com 1 frase
        resp_metricas = controlador.processar_mensagem(1, "/metricas")
        assert "RELATÓRIO DE MÉTRICAS PEDAGÓGICAS" in resp_metricas.texto
        assert "horario_funcionamento" in resp_metricas.texto


def test_formatacao_html_segura_e_divisao_telegram():
    texto_original = "Olá **Aluno**! Intenção: `conceito_pln_historico` e *destaque* <tag>."
    html_gerado = formatar_para_html_telegram(texto_original)
    assert "<b>Aluno</b>" in html_gerado
    assert "<code>conceito_pln_historico</code>" in html_gerado
    assert "<i>destaque</i>" in html_gerado
    assert "&lt;tag&gt;" in html_gerado

    texto_longo = "\n".join([f"Linha de teste número {i}" for i in range(300)])
    partes = dividir_mensagem_longa(texto_longo, limite=1000)
    assert len(partes) > 1
    assert all(len(p) <= 1000 for p in partes)
