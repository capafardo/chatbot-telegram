"""Módulo de Configuração e Compatibilidade Multiplataforma.

Referência Didática:
    - Boas práticas de Engenharia de Software e Segurança em Chatbots (Aulas 07 e 08).
    - Compatibilidade total entre Linux (Debian/Ubuntu) e Windows 11.
    - Isolamento de credenciais usando variáveis de ambiente (.env).
    - Suporte a customização de dataset (DATASET_PATH) e radicalização (USE_STEMMING).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Set

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
DIRETORIO_BASE: Path = Path(__file__).resolve().parent.parent

ARQUIVO_ENV: Path = DIRETORIO_BASE / ".env"
ARQUIVO_ENV_EXEMPLO: Path = DIRETORIO_BASE / ".env.example"

# Carregamento antecipado das variáveis de ambiente do .env
if load_dotenv and ARQUIVO_ENV.exists():
    load_dotenv(dotenv_path=ARQUIVO_ENV, encoding="utf-8")

DIRETORIO_DADOS: Path = DIRETORIO_BASE / "data"

# Permite que um aluno aponte para outro arquivo JSON de intenções via .env
_caminho_dataset_env: str = os.getenv("DATASET_PATH", "").strip()
if _caminho_dataset_env:
    _caminho_resolvido = Path(_caminho_dataset_env)
    ARQUIVO_INTENCOES: Path = (
        _caminho_resolvido
        if _caminho_resolvido.is_absolute()
        else (DIRETORIO_BASE / _caminho_resolvido)
    )
else:
    ARQUIVO_INTENCOES: Path = DIRETORIO_DADOS / "intencoes.json"

ARQUIVO_GIRIAS: Path = DIRETORIO_DADOS / "dicionario_girias.json"
DIRETORIO_BACKUP: Path = DIRETORIO_DADOS / "backups"

DIRETORIO_LOGS: Path = DIRETORIO_BASE / "logs"
ARQUIVO_LOG_JSONL: Path = DIRETORIO_LOGS / "conversas.jsonl"
ARQUIVO_LOG_CSV: Path = DIRETORIO_LOGS / "conversas.csv"

# Garante que as pastas necessárias existam em tempo de execução
DIRETORIO_DADOS.mkdir(parents=True, exist_ok=True)
DIRETORIO_BACKUP.mkdir(parents=True, exist_ok=True)
DIRETORIO_LOGS.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. LEITURA DE PARÂMETROS DE AMBIENTE
# ==============================================================================
# Chave da API do Telegram
CHAVE_API_TELEGRAM: str = os.getenv("TELEGRAM_API_KEY", "").strip()

# Lista de permissão (Whitelist) de administradores (chat_ids autorizados no /regenerar)
_lista_bruta: str = os.getenv("TELEGRAM_WHITELIST", "").strip()
LISTA_PERMISSAO_TELEGRAM: Set[str] = {
    item.strip() for item in _lista_bruta.split(",") if item.strip()
}

# Limiar de confiança para acionar o Fallback (padrão: 0.35)
try:
    LIMIAR_CONFIANCA: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
except ValueError:
    LIMIAR_CONFIANCA = 0.35

# Salt secreto para anonimização em conformidade com a LGPD (SHA-256)
SAL_ANONIMIZACAO: str = os.getenv(
    "SALT_ANONYMIZATION", "cetam_ibc_pln_2026_salt_seguro"
).strip()

# Modelo clássico padrão de ML: LogisticRegression ou MultinomialNB
MODELO_ML_PADRAO: str = os.getenv("DEFAULT_ML_MODEL", "LogisticRegression").strip()

# Uso de Stemming (radicalização) na vetorização TF-IDF (padrão: False)
_usar_stemming_bruto: str = os.getenv("USE_STEMMING", "false").strip().lower()
USAR_STEMMING: bool = _usar_stemming_bruto in ("true", "1", "sim", "yes", "on")

# Nível de log do sistema
NIVEL_LOG: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()

# ==============================================================================
# 3. DETECÇÃO E DIAGNÓSTICO DE AMBIENTE
# ==============================================================================
SISTEMA_OPERACIONAL_ATUAL: str = platform.system()
EH_WINDOWS: bool = SISTEMA_OPERACIONAL_ATUAL.lower() == "windows"
EH_LINUX: bool = SISTEMA_OPERACIONAL_ATUAL.lower() == "linux"


def eh_usuario_admin(id_usuario: int | str) -> bool:
    """Verifica se um determinado ID de usuário está na lista administrativa (Whitelist).

    Regra Didática:
        Se a lista estiver vazia, nenhum usuário remoto tem privilégios
        de curadoria (/regenerar) pelo Telegram por motivos de segurança,
        restringindo esse comando ao console/CLI local do professor.
        Se a lista contiver IDs, apenas os IDs listados têm permissão.
    """
    if not LISTA_PERMISSAO_TELEGRAM:
        return False
    return str(id_usuario) in LISTA_PERMISSAO_TELEGRAM


def validar_config_telegram() -> tuple[bool, str]:
    """Valida se o ambiente possui os pré-requisitos para iniciar o bot do Telegram."""
    if not CHAVE_API_TELEGRAM or "sua_chave_aqui" in CHAVE_API_TELEGRAM:
        return (
            False,
            "TELEGRAM_API_KEY não configurada no arquivo .env. "
            "Para rodar no Telegram, configure seu token do @BotFather. "
            "Você também pode rodar localmente no terminal com o modo CLI.",
        )
    return True, "Configuração do Telegram validada com sucesso."


def obter_resumo_sistema() -> dict[str, str | bool | float | int]:
    """Retorna um resumo de informações do ambiente operacional e de configuração."""
    return {
        "sistema_operacional": SISTEMA_OPERACIONAL_ATUAL,
        "eh_windows": EH_WINDOWS,
        "eh_linux": EH_LINUX,
        "diretorio_base": str(DIRETORIO_BASE),
        "intencoes_existe": ARQUIVO_INTENCOES.exists(),
        "girias_existe": ARQUIVO_GIRIAS.exists(),
        "limiar_fallback": LIMIAR_CONFIANCA,
        "usar_stemming": USAR_STEMMING,
        "tamanho_whitelist": len(LISTA_PERMISSAO_TELEGRAM),
        "telegram_configurado": bool(
            CHAVE_API_TELEGRAM and "sua_chave_aqui" not in CHAVE_API_TELEGRAM
        ),
    }
