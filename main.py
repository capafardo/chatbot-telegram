"""
CETAM - CENTRO DE EDUCAÇÃO TECNOLÓGICA DO AMAZONAS
IBC - INSTITUTO BENJAMIN CONSTANT
CURSO TÉCNICO EM INTELIGÊNCIA ARTIFICIAL
Processamento de Linguagem Natural - PLN e Chatbots
Prof. Agnaldo Marcelo Fournier
Exemplo de chatbot tutor pln com integração ao Telegram
=======================================================

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

from src.adaptador_cli import AdaptadorCLI
from src.adaptador_telegram import AdaptadorTelegram
from src.configuracao import validar_config_telegram
from src.controlador_bot import ControladorBot


def ler_argumentos() -> argparse.Namespace:
    """Configura e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Assistente Virtual Educacional com PLN Clássico (Telegram / CLI)"
    )
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument(
        "--cli", "-c",
        action="store_true",
        help="Executa o chatbot localmente no terminal (modo contingência de laboratório)."
    )
    grupo.add_argument(
        "--telegram", "-t",
        action="store_true",
        help="Executa o chatbot conectado ao canal do Telegram (modo polling)."
    )
    grupo.add_argument(
        "--metricas", "-m",
        action="store_true",
        help="Calcula e exibe as métricas de validação cruzada do modelo e dados de uso."
    )
    grupo.add_argument(
        "--regenerar", "-r",
        action="store_true",
        help="Inicia o painel interativo de curadoria de mensagens para o auto-aperfeiçoamento."
    )
    return parser.parse_args()


def principal() -> None:
    """Função principal de inicialização do sistema."""
    args = ler_argumentos()
    controlador = ControladorBot()

    # Modo 1: Apenas exibir métricas e sair
    if args.metricas:
        print("\n" + controlador.avaliador.formatar_relatorio_texto() + "\n")
        sys.exit(0)

    # Modo 2: Apenas abrir o painel de curadoria supervisionada
    if args.regenerar:
        cli = AdaptadorCLI(controlador)
        cli.executar_regenerar_interativo()
        sys.exit(0)

    # Modo 3: Terminal forçado (--cli)
    if args.cli:
        cli = AdaptadorCLI(controlador)
        cli.iniciar()
        sys.exit(0)

    # Modo 4: Telegram forçado (--telegram)
    if args.telegram:
        valido, msg = validar_config_telegram()
        if not valido:
            print(f"\n❌ Erro de Configuração: {msg}\n")
            sys.exit(1)
        adaptador = AdaptadorTelegram(controlador)
        adaptador.iniciar_polling()
        sys.exit(0)

    # Modo Padrão: Seleção Inteligente
    valido, _ = validar_config_telegram()
    if valido:
        print("Chave do Telegram detectada no .env. Iniciando bot do Telegram...")
        print("(Dica didática: para rodar no terminal sem Telegram, use: python main.py --cli)")
        adaptador = AdaptadorTelegram(controlador)
        adaptador.iniciar_polling()
    else:
        print("TELEGRAM_API_KEY não configurada no .env.")
        print("Iniciando automaticamente o modo Terminal (CLI) para testes locais...")
        cli = AdaptadorCLI(controlador)
        cli.iniciar()


if __name__ == "__main__":
    principal()
