"""Ponto de Entrada Principal do Chatbot Tutor de PLN & Chatbots (CETAM / IBC).

Referência Didática:
    - Inicialização unificada compatível com Linux (Debian/Ubuntu) e Windows 11.
    - Suporte a múltiplos modos de operação:
        1. Telegram Bot (modo prioritário via Polling);
        2. Terminal CLI (modo de contingência para laboratório);
        3. Avaliação direta de Métricas (--metricas);
        4. Painel de Curadoria Supervisionada (--regenerar).

Uso:
    python main.py             # Inicia Telegram (se configurado) ou CLI automaticamente
    python main.py --cli       # Força execução no terminal
    python main.py --telegram  # Força inicialização do bot no Telegram
    python main.py --metricas  # Exibe relatório de acurácia, F1 e matriz de confusão
    python main.py --regenerar # Inicia painel de curadoria do dataset
"""

from __future__ import annotations

import argparse
import sys

from src.controlador_bot import BotController
from src.adaptador_cli import CLIAdapter
from src.configuracao import get_system_summary, validate_telegram_config
from src.adaptador_telegram import TelegramAdapter


def parse_arguments() -> argparse.Namespace:
    """Configura e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Chatbot Tutor de PLN & Chatbots — CETAM / IBC (Manaus/AM)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cli", "-c",
        action="store_true",
        help="Executa o chatbot localmente no terminal (modo contingência de laboratório)."
    )
    group.add_argument(
        "--telegram", "-t",
        action="store_true",
        help="Executa o chatbot conectado ao canal do Telegram (modo polling)."
    )
    group.add_argument(
        "--metricas", "-m",
        action="store_true",
        help="Calcula e exibe as métricas de validação cruzada do modelo e dados de uso."
    )
    group.add_argument(
        "--regenerar", "-r",
        action="store_true",
        help="Inicia o painel interativo de curadoria de mensagens para o auto-aperfeiçoamento."
    )
    return parser.parse_args()


def main() -> None:
    """Função de inicialização do sistema."""
    args = parse_arguments()
    controller = BotController()

    # Modo 1: Apenas exibir métricas e sair
    if args.metricas:
        print("\n" + controller.evaluator.format_report_text() + "\n")
        sys.exit(0)

    # Modo 2: Apenas abrir o painel de curadoria supervisionada
    if args.regenerar:
        cli = CLIAdapter(controller)
        cli.run_regenerate_interactive()
        sys.exit(0)

    # Modo 3: Terminal forçado (--cli)
    if args.cli:
        cli = CLIAdapter(controller)
        cli.start()
        sys.exit(0)

    # Modo 4: Telegram forçado (--telegram)
    if args.telegram:
        valid, msg = validate_telegram_config()
        if not valid:
            print(f"\n❌ Erro de Configuração: {msg}\n")
            sys.exit(1)
        adapter = TelegramAdapter(controller)
        adapter.start_polling()
        sys.exit(0)

    # Modo Padrão: Seleção Inteligente
    valid, _ = validate_telegram_config()
    if valid:
        print("Chave do Telegram detectada no .env. Iniciando bot do Telegram...")
        print("(Dica didática: para rodar no terminal sem Telegram, use: python main.py --cli)")
        adapter = TelegramAdapter(controller)
        adapter.start_polling()
    else:
        print("TELEGRAM_API_KEY não configurada no .env.")
        print("Iniciando automaticamente o modo Terminal (CLI) para testes locais...")
        cli = CLIAdapter(controller)
        cli.start()


if __name__ == "__main__":
    main()
