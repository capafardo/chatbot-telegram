"""Adaptador de Linha de Comando (CLI) para Contingência e Uso em Laboratório.

Referência Didática:
    - Momento 4 das Aulas 07, 08 e 10 do Plano de Ensino (CETAM / IBC).
    - Plano de contingência: simulação local no terminal do computador caso ocorram
      restrições de internet ou bloqueio do Telegram na rede da escola.
    - Oferece menu interativo completo de administração e curadoria supervisionada (/regenerar).
    - Modular: exibe o nome e identidade do bot configurados em `data/intencoes.json`.
"""

from __future__ import annotations

from src.configuracao import obter_resumo_sistema
from src.controlador_bot import ControladorBot


class AdaptadorCLI:
    """Interface de terminal para interação com o assistente e curadoria pelo professor."""

    def __init__(self, controlador: ControladorBot | None = None) -> None:
        self.controlador: ControladorBot = controlador or ControladorBot()

    def executar_regenerar_interativo(self) -> None:
        """Menu interativo no terminal para curadoria supervisionada (/regenerar)."""
        print("\n" + "=" * 65)
        print("🛠️  PAINEL DE CURADORIA SUPERVISIONADA (/regenerar)")
        print("=" * 65)

        pendentes = self.controlador.aperfeicoador.obter_fallbacks_pendentes()
        if not pendentes:
            print("✅ Nenhuma mensagem com Fallback pendente de revisão!")
            print("=" * 65 + "\n")
            return

        todas_intencoes = self.controlador.ml.obter_todas_intencoes()
        print(f"Foram encontradas {len(pendentes)} mensagem(ns) pendente(s):\n")

        for idx, item in enumerate(pendentes, start=1):
            frase = item["mensagem_anonimizada"]
            id_inter = item["id_interacao"]

            print(f"\n[{idx}/{len(pendentes)}] Mensagem Recebida: \"{frase}\"")
            print("Opções:")
            print("  [1] Associar a uma intenção existente")
            print("  [2] Criar uma nova intenção no dataset")
            print("  [3] Ignorar / Descartar (ruído/fora de escopo)")
            print("  [0] Sair do modo regenerar")

            escolha = input("Escolha uma opção (0-3): ").strip()
            if escolha == "0":
                print("Curadoria encerrada.")
                break

            elif escolha == "1":
                print("\nIntenções disponíveis:")
                for i_idx, tag in enumerate(todas_intencoes, start=1):
                    print(f"  {i_idx:2d}. {tag}")

                selecao = input("\nDigite o nome ou número da intenção: ").strip()
                tag_alvo = ""
                if selecao.isdigit() and 1 <= int(selecao) <= len(todas_intencoes):
                    tag_alvo = todas_intencoes[int(selecao) - 1]
                elif selecao in todas_intencoes:
                    tag_alvo = selecao
                else:
                    print("❌ Intenção inválida. Operação cancelada para esta mensagem.")
                    continue

                _, msg = self.controlador.aperfeicoador.adicionar_frase_a_intencao(
                    tag_intencao=tag_alvo,
                    frase=frase,
                    id_interacao=id_inter
                )
                print(f"Resultado: {msg}")

            elif escolha == "2":
                tag = input("Tag da nova intenção (ex: regras_laboratorio): ").strip()
                desc = input("Descrição curta da intenção: ").strip()
                resp = input("Resposta que o robô deve enviar: ").strip()

                if not tag or not resp:
                    print("❌ Tag e resposta são obrigatórios.")
                    continue

                _, msg = self.controlador.aperfeicoador.criar_nova_intencao(
                    tag=tag,
                    categoria="tecnica",
                    descricao=desc,
                    primeiro_padrao=frase,
                    primeira_resposta=resp,
                    id_interacao=id_inter
                )
                print(f"Resultado: {msg}")
                # Atualiza lista de intenções
                todas_intencoes = self.controlador.ml.obter_todas_intencoes()

            elif escolha == "3":
                self.controlador.aperfeicoador.marcar_log_revisado(id_inter)
                print("Mensagem marcada como revisada/descartada.")

        print("\n" + "=" * 65 + "\n")

    def iniciar(self) -> None:
        """Inicia o loop interativo no terminal."""
        resumo = obter_resumo_sistema()
        ident = self.controlador.obter_identidade_bot()
        nome_exibicao = ident["nome_bot"].upper()
        prompt_bot = ident["nome_curto"]
        instituicao = ident["instituicao"] or "Modo Local"

        print("\n" + "=" * 65)
        print(f"🤖 {nome_exibicao} — MODO TERMINAL (CLI)")
        print(f"🏛️  {instituicao} | SO: {resumo['sistema_operacional']}")
        print("💡 Digite sua mensagem, use comandos (/debug, /metricas, /ajuda)")
        print("   ou digite 'sair' para encerrar.")
        print("=" * 65 + "\n")

        id_usuario = "console_local"

        while True:
            try:
                entrada_usuario = input("\nVocê > ").strip()
                if not entrada_usuario:
                    continue

                if entrada_usuario.lower() in ("sair", "exit", "quit", "q"):
                    print("\nEncerrando sessão no terminal. Bons estudos e até logo!")
                    break

                if entrada_usuario == "/regenerar":
                    self.executar_regenerar_interativo()
                    continue

                # Processa via controlador central
                resposta = self.controlador.processar_mensagem(
                    id_usuario=id_usuario,
                    mensagem=entrada_usuario,
                    canal="cli"
                )

                print(f"\n{prompt_bot} >\n{resposta.texto}")

            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando o assistente...")
                break
            except Exception as erro:
                print(f"\n❌ Erro durante o processamento: {erro}")
