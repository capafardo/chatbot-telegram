"""Testes para o módulo de anonimização e conformidade com a LGPD."""

import tempfile
from pathlib import Path
from src.registrador_privacidade import PrivacyLogger


def test_pii_sanitization():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl = Path(tmpdir) / "test.jsonl"
        csv = Path(tmpdir) / "test.csv"
        logger = PrivacyLogger(jsonl_path=jsonl, csv_path=csv)

        text = "Meu CPF é 012.345.678-90, zap 92 98765-4321 e email teste@cetam.am.gov.br"
        sanitized = logger.sanitize_message(text)

        assert "[CPF_OCULTO]" in sanitized
        assert "[TELEFONE_OCULTO]" in sanitized
        assert "[EMAIL_OCULTO]" in sanitized
        assert "012.345.678-90" not in sanitized
        assert "teste@cetam.am.gov.br" not in sanitized


def test_user_id_anonymization():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl = Path(tmpdir) / "test.jsonl"
        csv = Path(tmpdir) / "test.csv"
        logger = PrivacyLogger(jsonl_path=jsonl, csv_path=csv, salt="salt_teste")

        user_anon_1 = logger.anonymize_user_id(123456)
        user_anon_2 = logger.anonymize_user_id(123456)
        user_anon_3 = logger.anonymize_user_id(999999)

        assert user_anon_1.startswith("usr_")
        assert user_anon_1 == user_anon_2
        assert user_anon_1 != user_anon_3
        assert "123456" not in user_anon_1
