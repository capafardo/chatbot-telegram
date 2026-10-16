"""Testes de integração para o BotController."""

from src.bot_controller import BotController


def test_start_command():
    controller = BotController()
    response = controller.handle_message(1001, "/start")
    assert "CETAM" in response.text
    assert "Agnaldo Marcelo Pereira Fournier" in response.text
    assert "LGPD" in response.text


def test_debug_toggle_and_response():
    controller = BotController()
    user_id = 2002

    # Liga debug
    resp_toggle = controller.handle_message(user_id, "/debug")
    assert "ATIVADO" in resp_toggle.text
    assert controller.is_debug_active(user_id)

    # Pergunta normal com debug ativo
    resp_question = controller.handle_message(user_id, "O que é NLP?")
    assert "MODO DEBUG — RAIO-X DO PIPELINE DE PLN" in resp_question.text
    assert "conceito_pln_historico" in resp_question.text
    assert "Resposta do Tutor" in resp_question.text

    # Desliga debug
    resp_off = controller.handle_message(user_id, "/debug")
    assert "DESATIVADO" in resp_off.text
    assert not controller.is_debug_active(user_id)


def test_regenerar_access_control():
    controller = BotController()
    # Usuário sem whitelist no Telegram
    resp_unauth = controller.handle_message(9999, "/regenerar", channel="telegram")
    assert resp_unauth.action_type == "unauthorized"
    assert "Acesso Restrito" in resp_unauth.text

    # No canal CLI local, o acesso é liberado para o operador
    resp_cli = controller.handle_message("console", "/regenerar", channel="cli")
    assert resp_cli.action_type in ("regenerate_flow", "text")
