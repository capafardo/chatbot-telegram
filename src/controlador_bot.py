"""Controlador Central do Chatbot (Orquestrador Agnóstico de Canal e Tema).

Referência Didática:
    - Aulas 07 e 08 do Plano de Ensino (CETAM / IBC).
    - Separação entre a Lógica de Negócio (PLN, ML, Logs, Regras) e os canais
      de comunicação (Telegram e Terminal CLI).
    - Arquitetura Modular ("White-Label"): textos de apresentação, exemplos de
      ajuda e nome do assistente são carregados dinamicamente dos metadados de
      `data/intencoes.json`, permitindo que alunos criem chatbots para outros
      assuntos apenas editando o JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from src.auto_aperfeicoamento import AutoAperfeicoamento
from src.avaliador import AvaliadorModelo
from src.configuracao import LIMIAR_CONFIANCA, eh_usuario_admin
from src.motor_ml import MotorML, ResultadoPredicao
from src.pipeline_pln import PipelinePLN
from src.registrador_privacidade import RegistradorPrivacidade


@dataclass
class RespostaBot:
    """Resposta estruturada do bot para ser renderizada pelo canal (Telegram ou CLI)."""
    texto: str
    eh_debug: bool = False
    info_debug: Optional[Dict[str, Any]] = None
    predicao: Optional[ResultadoPredicao] = None
    tipo_acao: str = "texto"  # 'texto', 'menu', 'fluxo_regenerar', 'nao_autorizado'


class ControladorBot:
    """Orquestrador central de regras, pipeline de PLN, ML e persistência."""

    def __init__(
        self,
        pipeline_pln: PipelinePLN | None = None,
        motor_ml: MotorML | None = None,
        registrador_privacidade: RegistradorPrivacidade | None = None
    ) -> None:
        self.pln: PipelinePLN = pipeline_pln or PipelinePLN()
        self.ml: MotorML = motor_ml or MotorML(pipeline_pln=self.pln)
        self.registrador: RegistradorPrivacidade = (
            registrador_privacidade or RegistradorPrivacidade()
        )
        self.avaliador: AvaliadorModelo = AvaliadorModelo(
            motor_ml=self.ml, registrador_privacidade=self.registrador
        )
        self.aperfeicoador: AutoAperfeicoamento = AutoAperfeicoamento(
            motor_ml=self.ml,
            registrador_privacidade=self.registrador,
            caminho_intencoes=self.ml.caminho_intencoes,
        )

        # Sessões com modo /debug ativado (rastreado pelo ID do usuário)
        self.usuarios_debug_ativos: Set[str] = set()

    def debug_esta_ativo(self, id_usuario: int | str) -> bool:
        """Verifica se o usuário atual está com o modo /debug habilitado."""
        return str(id_usuario) in self.usuarios_debug_ativos

    def alternar_debug(self, id_usuario: int | str) -> bool:
        """Alterna o status do modo /debug (liga/desliga) para um usuário."""
        id_str = str(id_usuario)
        if id_str in self.usuarios_debug_ativos:
            self.usuarios_debug_ativos.remove(id_str)
            return False
        self.usuarios_debug_ativos.add(id_str)
        return True

    def obter_identidade_bot(self) -> Dict[str, Any]:
        """Recupera informações de identidade e personalização do bot a partir do JSON."""
        meta = self.ml.obter_metadados_bot()
        nome_bot = meta.get("nome_bot") or "Chatbot Assistente Educacional"
        nome_curto = meta.get("nome_curto") or "Tutor"
        dominio = (
            meta.get("dominio_assunto")
            or meta.get("componente_curricular")
            or "atendimento e dúvidas"
        )
        instituicao = meta.get("instituicao") or ""
        escola = meta.get("escola") or ""
        curso = meta.get("curso") or ""
        docente = meta.get("docente") or ""
        mensagem_boas_vindas = meta.get("mensagem_boas_vindas") or ""
        exemplos: List[str] = meta.get("exemplos_perguntas") or []

        return {
            "nome_bot": str(nome_bot),
            "nome_curto": str(nome_curto),
            "dominio": str(dominio),
            "instituicao": str(instituicao),
            "escola": str(escola),
            "curso": str(curso),
            "docente": str(docente),
            "mensagem_boas_vindas": str(mensagem_boas_vindas),
            "exemplos_perguntas": [str(e) for e in exemplos],
        }

    def _construir_mensagem_boas_vindas(self) -> str:
        """Monta dinamicamente a mensagem do comando /start usando os metadados do JSON."""
        ident = self.obter_identidade_bot()
        if ident["mensagem_boas_vindas"]:
            return ident["mensagem_boas_vindas"]

        linhas = [f"👋 **Olá! Bem-vindo ao {ident['nome_bot']}!**\n"]

        bloco_institucional = []
        if ident["instituicao"]:
            inst_txt = ident["instituicao"]
            if ident["escola"]:
                inst_txt += f" — {ident['escola']}"
            bloco_institucional.append(f"🏛️ **Instituição:** {inst_txt}")
        if ident["curso"]:
            bloco_institucional.append(f"🎓 **Curso:** {ident['curso']}")
        if ident["docente"]:
            bloco_institucional.append(f"👨‍🏫 **Docente:** {ident['docente']}")

        if bloco_institucional:
            linhas.append("\n".join(bloco_institucional) + "\n")

        linhas.extend([
            "🤖 Sou um assistente construído com **Machine Learning clássico supervisionado "
            "(TF-IDF + Regressão Logística / Naive Bayes)**, sem uso de LLMs.\n",
            "🔒 **Aviso de Privacidade (LGPD):** Esta conversa tem finalidade pedagógica. "
            "Seu ID e eventuais dados pessoais são anonimizados antes do registro em log.\n",
            "💡 **O que você pode fazer:**",
            f"• Fazer perguntas em português sobre **{ident['dominio']}**;",
            "• Digitar `/ajuda` para ver todos os comandos e exemplos de perguntas;",
            "• Digitar `/debug` para ver o raio-X matemático de como analiso cada frase!",
        ])
        return "\n".join(linhas)

    def _construir_mensagem_ajuda(self) -> str:
        """Monta dinamicamente a mensagem do comando /ajuda com exemplos vindos do JSON."""
        ident = self.obter_identidade_bot()
        linhas = [
            f"📖 **GUIA DE COMANDOS — {ident['nome_bot'].upper()}:**\n",
            "• `/start` - Apresentação do assistente e política de privacidade (LGPD);",
            "• `/ajuda` - Este menu de instruções e sugestões;",
            "• `/debug` - Liga/desliga o modo raio-X do pipeline de PLN para suas mensagens;",
            "• `/debug <frase>` - Inspeciona imediatamente os passos de PLN de uma frase avulsa;",
            "• `/metricas` - Exibe a acurácia, F1-Score e a matriz de confusão dos modelos;",
            "• `/regenerar` - (Apenas Administradores) Curadoria de dúvidas para o bot aprender!",
        ]

        exemplos = ident["exemplos_perguntas"]
        if exemplos:
            linhas.append("\n💬 **Exemplos de perguntas que você pode me fazer:**")
            for ex in exemplos:
                linhas.append(f"• *'{ex}'*")

        return "\n".join(linhas)

    def formatar_bloco_debug(self, texto: str, predicao: ResultadoPredicao) -> str:
        """Formata didaticamente o raio-X completo do pipeline de PLN da mensagem."""
        inspecao = self.pln.inspecionar(texto)
        candidatos_lr = predicao.principais_candidatos
        principais_termos = predicao.termos_tfidf_ativos[:5]
        status_stem = (
            "ATIVO na vetorização"
            if inspecao["7_stemming_ativo_no_modelo"]
            else "DEMONSTRATIVO (desativado no vetorizador)"
        )

        texto_debug = [
            "🔬 **[MODO DEBUG — RAIO-X DO PIPELINE DE PLN]**",
            f"1️⃣ Texto Original: \"{inspecao['1_texto_original']}\"",
            f"2️⃣ Normalização (minúsculas): \"{inspecao['2_normalizado']}\"",
            f"3️⃣ Expansão de Gírias: \"{inspecao['3_expansao_girias']}\"",
            f"4️⃣ Limpeza Regex (ruídos): \"{inspecao['4_limpeza_regex']}\"",
            f"5️⃣ Tokens Brutos: {inspecao['5_tokens_brutos']}",
            f"6️⃣ Sem Stopwords (negações mantidas): {inspecao['6_apos_stopwords']}",
            f"7️⃣ Radicais / Stemming ({status_stem}): {inspecao['7_radicais_stemming']}",
            f"8️⃣ Texto Final (Vetorizador): \"{inspecao['8_texto_final_vetorizador']}\"",
            f"9️⃣ Termos de Maior Peso TF-IDF: {principais_termos if principais_termos else 'Nenhum termo do vocabulário'}",
            "--------------------------------------------------",
            f"🤖 Modelo Primário: **{predicao.modelo_primario}**",
            f"🎯 Intenção Predita: `{predicao.intencao}`",
            f"📈 Certeza: **{predicao.confianca * 100:.1f}%** (Limiar Fallback: {LIMIAR_CONFIANCA * 100:.1f}%)",
            f"🛡️ Fallback Acionado: **{'SIM' if predicao.acionou_fallback else 'NÃO'}**",
            "📊 Top 3 Hipóteses:",
        ]
        for posicao, (intencao, prob) in enumerate(candidatos_lr, start=1):
            texto_debug.append(f"   {posicao}. `{intencao}` ({prob * 100:.1f}%)")

        texto_debug.append(
            f"💡 Predição Comparativa (Naive Bayes): `{predicao.intencao_nb}` ({predicao.confianca_nb * 100:.1f}%)"
        )
        texto_debug.append("==================================================")
        return "\n".join(texto_debug)

    def processar_mensagem(
        self,
        id_usuario: int | str,
        mensagem: str,
        canal: str = "telegram"
    ) -> RespostaBot:
        """Processa comandos ou perguntas normais e retorna a resposta formatada."""
        texto_limpo = mensagem.strip()

        # ----------------------------------------------------------------------
        # COMANDO: /start
        # ----------------------------------------------------------------------
        if texto_limpo.startswith("/start"):
            return RespostaBot(
                texto=self._construir_mensagem_boas_vindas(),
                tipo_acao="texto"
            )

        # ----------------------------------------------------------------------
        # COMANDO: /ajuda ou /help
        # ----------------------------------------------------------------------
        if texto_limpo.startswith(("/ajuda", "/help")):
            return RespostaBot(
                texto=self._construir_mensagem_ajuda(),
                tipo_acao="texto"
            )

        # ----------------------------------------------------------------------
        # COMANDO: /debug ou /debug <frase>
        # ----------------------------------------------------------------------
        if texto_limpo.startswith("/debug"):
            argumentos = texto_limpo[6:].strip()
            if argumentos:
                predicao = self.ml.prever(argumentos)
                saida_debug = self.formatar_bloco_debug(argumentos, predicao)
                return RespostaBot(texto=saida_debug, eh_debug=True, tipo_acao="texto")
            else:
                novo_estado = self.alternar_debug(id_usuario)
                status_str = "🟢 **ATIVADO**" if novo_estado else "🔴 **DESATIVADO**"
                msg = (
                    f"Modo Debug {status_str} para a sua sessão!\n"
                    f"{'A partir de agora, cada resposta virá acompanhada do raio-X do pipeline de PLN.' if novo_estado else 'As próximas respostas voltarão ao modo normal de atendimento.'}"
                )
                return RespostaBot(texto=msg, eh_debug=novo_estado, tipo_acao="texto")

        # ----------------------------------------------------------------------
        # COMANDO: /metricas
        # ----------------------------------------------------------------------
        if texto_limpo.startswith("/metricas"):
            relatorio = self.avaliador.formatar_relatorio_texto()
            return RespostaBot(texto=relatorio, tipo_acao="texto")

        # ----------------------------------------------------------------------
        # COMANDO: /regenerar (Controle de Acesso Administrativo)
        # ----------------------------------------------------------------------
        if texto_limpo.startswith("/regenerar"):
            # No canal CLI local o acesso é irrestrito (professor no console);
            # No Telegram, exige que o ID esteja na TELEGRAM_WHITELIST do .env
            if canal == "telegram" and not eh_usuario_admin(id_usuario):
                recusa = (
                    "⛔ **Acesso Restrito:**\n"
                    "O comando `/regenerar` modifica o dataset de treinamento do robô e é restrito "
                    "aos administradores cadastrados na `TELEGRAM_WHITELIST` do `.env`.\n"
                    "Realize a curadoria pelo terminal local (`python main.py --regenerar`)!"
                )
                return RespostaBot(texto=recusa, tipo_acao="nao_autorizado")

            pendentes = self.aperfeicoador.obter_fallbacks_pendentes()
            if not pendentes:
                msg = (
                    "🎉 **Nenhuma mensagem pendente de curadoria!**\n"
                    "Todas as dúvidas em que o robô caiu no Fallback já foram revisadas ou não há novos registros no log."
                )
                return RespostaBot(texto=msg, tipo_acao="texto")

            linhas = [
                "📋 **CURADORIA SUPERVISIONADA (/regenerar)**",
                f"Há **{len(pendentes)}** mensagem(ns) anonimizada(s) com baixa confiança aguardando revisão:\n",
            ]
            for idx, item in enumerate(pendentes[:5], start=1):
                linhas.append(
                    f"{idx}. *\"{item['mensagem_anonimizada']}\"* (Data: {item['data']} {item['hora']})"
                )

            linhas.append(
                "\n💡 Para vincular cada frase a uma intenção existente ou criar novas intenções "
                "com segurança, execute no terminal local:\n`python main.py --regenerar`"
            )
            return RespostaBot(texto="\n".join(linhas), tipo_acao="fluxo_regenerar")

        # ----------------------------------------------------------------------
        # FLUXO PADRÃO: INFERÊNCIA DE PERGUNTAS DO USUÁRIO
        # ----------------------------------------------------------------------
        predicao = self.ml.prever(texto_limpo)

        # Registro persistente e anonimizado em conformidade com a LGPD
        self.registrador.registrar_interacao(
            id_usuario_bruto=id_usuario,
            mensagem_bruta=texto_limpo,
            intencao_predita=predicao.intencao,
            confianca=predicao.confianca,
            acionou_fallback=predicao.acionou_fallback,
            canal=canal
        )

        # Se o modo debug estiver ativo, anexa o bloco técnico antes da resposta
        texto_final = predicao.resposta
        if self.debug_esta_ativo(id_usuario):
            ident = self.obter_identidade_bot()
            bloco_debug = self.formatar_bloco_debug(texto_limpo, predicao)
            texto_final = (
                f"{bloco_debug}\n\n💬 **Resposta do {ident['nome_curto']}:**\n{predicao.resposta}"
            )

        return RespostaBot(
            texto=texto_final,
            eh_debug=self.debug_esta_ativo(id_usuario),
            predicao=predicao,
            tipo_acao="texto"
        )
