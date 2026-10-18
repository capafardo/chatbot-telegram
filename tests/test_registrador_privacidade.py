"""Testes para o módulo de anonimização e conformidade com a LGPD (RegistradorPrivacidade)."""

import csv
import tempfile
from pathlib import Path

from src.registrador_privacidade import RegistradorPrivacidade


def test_sanitizacao_pii():
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_jsonl = Path(tmpdir) / "teste.jsonl"
        caminho_csv = Path(tmpdir) / "teste.csv"
        registrador = RegistradorPrivacidade(caminho_jsonl=caminho_jsonl, caminho_csv=caminho_csv)

        texto = "Meu CPF é 012.345.678-90, zap 92 98765-4321 e email teste@cetam.am.gov.br"
        sanitizado = registrador.sanitizar_mensagem(texto)

        assert "[CPF_OCULTO]" in sanitizado
        assert "[TELEFONE_OCULTO]" in sanitizado
        assert "[EMAIL_OCULTO]" in sanitizado
        assert "012.345.678-90" not in sanitizado
        assert "teste@cetam.am.gov.br" not in sanitizado


def test_regex_nome_nao_apaga_frase_comum_do_aluno():
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_jsonl = Path(tmpdir) / "teste.jsonl"
        caminho_csv = Path(tmpdir) / "teste.csv"
        registrador = RegistradorPrivacidade(caminho_jsonl=caminho_jsonl, caminho_csv=caminho_csv)

        # Frase comum em minúsculas após "sou o" NÃO deve ser apagada
        frase_aluno = "oi prof, sou o aluno que faltou na aula de ontem e queria saber sobre tf-idf"
        resultado_aluno = registrador.sanitizar_mensagem(frase_aluno)
        assert "aluno que faltou na aula de ontem" in resultado_aluno
        assert "[NOME_OCULTO]" not in resultado_aluno

        # Nome próprio capitalizado DEVE ser ocultado preservando o restante da frase
        frase_com_nome = "Olá, meu nome é Carlos Eduardo da Silva e tenho uma dúvida"
        resultado_nome = registrador.sanitizar_mensagem(frase_com_nome)
        assert "[NOME_OCULTO]" in resultado_nome
        assert "Carlos Eduardo" not in resultado_nome
        assert "e tenho uma dúvida" in resultado_nome


def test_anonimizacao_id_usuario_e_sincronizacao_jsonl_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_jsonl = Path(tmpdir) / "teste.jsonl"
        caminho_csv = Path(tmpdir) / "teste.csv"
        registrador = RegistradorPrivacidade(
            caminho_jsonl=caminho_jsonl, caminho_csv=caminho_csv, sal="sal_teste"
        )

        usr_1 = registrador.anonimizar_id_usuario(123456)
        usr_2 = registrador.anonimizar_id_usuario(123456)
        usr_3 = registrador.anonimizar_id_usuario(999999)

        assert usr_1.startswith("usr_")
        assert usr_1 == usr_2
        assert usr_1 != usr_3
        assert "123456" not in usr_1

        # Registra interação e verifica sincronização ao marcar como revisada
        entrada = registrador.registrar_interacao(
            id_usuario_bruto=123456,
            mensagem_bruta="duvida desconhecida",
            intencao_predita="fallback",
            confianca=0.15,
            acionou_fallback=True,
            canal="cli"
        )
        id_interacao = entrada["id_interacao"]
        assert registrador.marcar_interacao_revisada(id_interacao)

        # Verifica no JSONL
        logs_jsonl = registrador.ler_todos_logs()
        assert logs_jsonl[0]["revisado_aprendizado"] is True

        # Verifica no CSV
        with open(caminho_csv, mode="r", encoding="utf-8") as f_csv:
            linhas_csv = list(csv.DictReader(f_csv))
            assert linhas_csv[0]["revisado_aprendizado"] == "True"
