"""Módulo de Configuração e Compatibilidade Multiplataforma.

Referência Didática:
    - Boas práticas de Engenharia de Software e Segurança em Chatbots (Aulas 07 e 08).
    - Compatibilidade total entre Linux (Debian/Ubuntu) e Windows 11.
    - Isolamento de credenciais usando variáveis de ambiente (.env).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import List, Set

# Tentativa segura de carregar o python-dotenv (se instalado no ambiente)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ==============================================================================
# 1. MAPEAMENTO DE DIRETÓRIOS E ARQUIVOS (PATHLIB AGNÓSTICO DE SO)
# ==============================================================================
# O uso de Path(__file__).resolve() garante caminhos absolutos corretos
# independentemente do diretório de onde o usuário executou o comando 'python'.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
INTENTS_FILE: Path = DATA_DIR / "intents.json"
SLANG_FILE: Path = DATA_DIR / "slang_dict.json"
BACKUP_DIR: Path = DATA_DIR / "backups"

LOGS_DIR: Path = BASE_DIR / "logs"
LOG_JSONL_FILE: Path = LOGS_DIR / "conversas.jsonl"
LOG_CSV_FILE: Path = LOGS_DIR / "conversas.csv"

ENV_FILE: Path = BASE_DIR / ".env"
ENV_EXAMPLE_FILE: Path = BASE_DIR / ".env.example"

# Garante que as pastas necessárias existam em tempo de execução
DATA_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CARREGAMENTO DE VARIÁVEIS DE AMBIENTE
# ==============================================================================
if load_dotenv and ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, encoding="utf-8")

# Configurações do Telegram
TELEGRAM_API_KEY: str = os.getenv("TELEGRAM_API_KEY", "").strip()

# Whitelist de usuários (chat_ids autorizados)
_raw_whitelist: str = os.getenv("TELEGRAM_WHITELIST", "").strip()
TELEGRAM_WHITELIST: Set[str] = {
    item.strip() for item in _raw_whitelist.split(",") if item.strip()
}

# Limiar de confiança para acionar o Fallback (padrão: 0.35)
try:
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
except ValueError:
    CONFIDENCE_THRESHOLD = 0.35

# Salt secreto para anonimização em conformidade com a LGPD (SHA-256)
SALT_ANONYMIZATION: str = os.getenv(
    "SALT_ANONYMIZATION", "cetam_ibc_pln_2026_salt_seguro"
).strip()

# Modelo clássico padrão de ML: LogisticRegression ou MultinomialNB
DEFAULT_ML_MODEL: str = os.getenv("DEFAULT_ML_MODEL", "LogisticRegression").strip()

# ==============================================================================
# 3. DETECÇÃO E DIAGNÓSTICO DE AMBIENTE
# ==============================================================================
CURRENT_OS: str = platform.system()
IS_WINDOWS: bool = CURRENT_OS.lower() == "windows"
IS_LINUX: bool = CURRENT_OS.lower() == "linux"


def is_user_admin(user_id: int | str) -> bool:
    """Verifica se um determinado ID de usuário está na Whitelist administrativa.

    Regra Didática:
        Se a Whitelist estiver vazia, nenhum usuário remoto tem privilégios
        de curadoria (/regenerar) pelo Telegram por motivos de segurança,
        restringindo esse comando ao console/CLI local do professor.
        Se a Whitelist contiver IDs, apenas os IDs listados têm permissão.
    """
    if not TELEGRAM_WHITELIST:
        return False
    return str(user_id) in TELEGRAM_WHITELIST


def validate_telegram_config() -> tuple[bool, str]:
    """Valida se o ambiente possui os pré-requisitos para iniciar o bot do Telegram."""
    if not TELEGRAM_API_KEY or "sua_chave_aqui" in TELEGRAM_API_KEY:
        return (
            False,
            "TELEGRAM_API_KEY não configurada no arquivo .env. "
            "Para rodar no Telegram, configure seu token do @BotFather. "
            "Você também pode rodar localmente no terminal com o modo CLI.",
        )
    return True, "Configuração do Telegram validada com sucesso."


def get_system_summary() -> dict[str, str | bool | float | int]:
    """Retorna um resumo de informações do ambiente operacional e de configuração."""
    return {
        "sistema_operacional": CURRENT_OS,
        "is_windows": IS_WINDOWS,
        "is_linux": IS_LINUX,
        "base_dir": str(BASE_DIR),
        "intents_existe": INTENTS_FILE.exists(),
        "slang_existe": SLANG_FILE.exists(),
        "limiar_fallback": CONFIDENCE_THRESHOLD,
        "whitelist_tamanho": len(TELEGRAM_WHITELIST),
        "telegram_configurado": bool(
            TELEGRAM_API_KEY and "sua_chave_aqui" not in TELEGRAM_API_KEY
        ),
    }
