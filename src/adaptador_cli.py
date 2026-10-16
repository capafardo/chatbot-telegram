"""Adaptador de Linha de Comando (CLI) para Contingência e Uso em Laboratório.

Referência Didática:
    - Momento 4 das Aulas 07, 08 e 10 do Plano de Ensino (CETAM / IBC).
    - Plano de contingência: simulação local no terminal do computador caso ocorram
      restrições de internet ou bloqueio do Telegram na rede da escola.
    - Oferece menu interativo completo de administração e curadoria supervisionada (/regenerar).
"""

from __future__ import annotations

import sys
from typing import Optional

from src.controlador_bot import BotController
from src.configuracao import get_system_summary


class CLIAdapter:
    """Interface de terminal para interação com o tutor e curadoria pelo professor."""

    def __init__(self, controller: BotController | None = None) -> None:
        self.controller: BotController = controller or BotController()

    def run_regenerate_interactive(self) -> None:
        """Menu interativo no terminal para curadoria supervisionada (/regenerar)."""
        print("\n" + "=" * 65)
        print("🛠️  PAINEL DE CURADORIA SUPERVISIONADA (/regenerar)")
        print("=" * 65)

        pending = self.controller.improver.get_pending_fallbacks()
        if not pending:
            print("✅ Nenhuma mensagem com Fallback pendente de revisão!")
            print("=" * 65 + "\n")
            return

        all_intents = self.controller.ml.get_all_intents()
        print(f"Foram encontradas {len(pending)} mensagem(ns) pendente(s):\n")

        for idx, item in enumerate(pending, start=1):
            phrase = item["mensagem_anonimizada"]
            inter_id = item["id_interacao"]

            print(f"\n[{idx}/{len(pending)}] Dúvida do Aluno: \"{phrase}\"")
            print("Opções:")
            print("  [1] Associar a uma intenção existente")
            print("  [2] Criar uma nova intenção no dataset")
            print("  [3] Ignorar / Descartar (ruído/fora de escopo)")
            print("  [0] Sair do modo regenerar")

            choice = input("Escolha uma opção (0-3): ").strip()
            if choice == "0":
                print("Curadoria encerrada.")
                break

            elif choice == "1":
                print("\nIntenções disponíveis:")
                for i_idx, tag in enumerate(all_intents, start=1):
                    print(f"  {i_idx:2d}. {tag}")

                sel = input("\nDigite o nome ou número da intenção: ").strip()
                target_tag = ""
                if sel.isdigit() and 1 <= int(sel) <= len(all_intents):
                    target_tag = all_intents[int(sel) - 1]
                elif sel in all_intents:
                    target_tag = sel
                else:
                    print("❌ Intenção inválida. Operação cancelada para esta mensagem.")
                    continue

                success, msg = self.controller.improver.add_phrase_to_intent(
                    intent_tag=target_tag,
                    phrase=phrase,
                    interaction_id=inter_id
                )
                print(f"Resultado: {msg}")

            elif choice == "2":
                tag = input("Tag da nova intenção (ex: regras_laboratorio): ").strip()
                desc = input("Descrição curta da intenção: ").strip()
                resp = input("Resposta que o robô deve enviar: ").strip()

                if not tag or not resp:
                    print("❌ Tag e resposta são obrigatórios.")
                    continue

                success, msg = self.controller.improver.create_new_intent(
                    tag=tag,
                    categoria="tecnica",
                    descricao=desc,
                    first_pattern=phrase,
                    first_response=resp,
                    interaction_id=inter_id
                )
                print(f"Resultado: {msg}")
                # Atualiza lista de intenções
                all_intents = self.controller.ml.get_all_intents()

            elif choice == "3":
                self.controller.improver.mark_log_reviewed(inter_id)
                print("Mensagem marcada como revisada/descartada.")

        print("\n" + "=" * 65 + "\n")

    def start(self) -> None:
        """Inicia o loop interativo no terminal."""
        summary = get_system_summary()
        print("\n" + "=" * 65)
        print("🤖 CHATBOT TUTOR DE PLN & CHATBOTS — MODO TERMINAL (CLI)")
        print(f"🏛️  CETAM / IBC | Sistema Operacional: {summary['sistema_operacional']}")
        print("💡 Digite sua mensagem, use comandos (/debug, /metricas, /ajuda)")
        print("   ou digite 'sair' para encerrar.")
        print("=" * 65 + "\n")

        user_id = "console_local"

        while True:
            try:
                user_input = input("\nVocê > ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("sair", "exit", "quit", "q"):
                    print("\nEncerrando sessão no terminal. Bons estudos e até logo!")
                    break

                if user_input == "/regenerar":
                    self.run_regenerate_interactive()
                    continue

                # Processa via controlador central
                response = self.controller.handle_message(
                    user_id=user_id,
                    message=user_input,
                    channel="cli"
                )

                print(f"\nTutor >\n{response.text}")

            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando o assistente...")
                break
            except Exception as e:
                print(f"\n❌ Erro durante o processamento: {e}")
