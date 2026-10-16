"""Controlador Central do Chatbot (Orquestrador Agnóstico de Canal).

Referência Didática:
    - Aulas 07 e 08 do Plano de Ensino (CETAM / IBC).
    - Separação entre a Lógica de Negócio (PLN, ML, Logs, Regras) e os canais
      de comunicação (Telegram e Terminal CLI).
    - Gerenciamento de sessões com o modo /debug ativo e controle de acesso a comandos administrativos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.configuracao import CONFIDENCE_THRESHOLD, is_user_admin
from src.avaliador import ModelEvaluator
from src.motor_ml import MLEngine, PredictionResult
from src.pipeline_pln import NLPPipeline
from src.registrador_privacidade import PrivacyLogger
from src.auto_aperfeicoamento import SelfImprover


@dataclass
class BotResponse:
    """Resposta estruturada do bot para ser renderizada pelo canal (Telegram ou CLI)."""
    text: str
    is_debug: bool = False
    debug_info: Optional[Dict[str, Any]] = None
    prediction: Optional[PredictionResult] = None
    action_type: str = "text"  # 'text', 'menu', 'regenerate_flow', 'unauthorized'


class BotController:
    """Orquestrador central de regras, pipeline de PLN, ML e persistência."""

    def __init__(
        self,
        nlp_pipeline: NLPPipeline | None = None,
        ml_engine: MLEngine | None = None,
        privacy_logger: PrivacyLogger | None = None
    ) -> None:
        self.nlp: NLPPipeline = nlp_pipeline or NLPPipeline()
        self.ml: MLEngine = ml_engine or MLEngine(nlp_pipeline=self.nlp)
        self.logger: PrivacyLogger = privacy_logger or PrivacyLogger()
        self.evaluator: ModelEvaluator = ModelEvaluator(
            ml_engine=self.ml, privacy_logger=self.logger
        )
        self.improver: SelfImprover = SelfImprover(
            ml_engine=self.ml, privacy_logger=self.logger
        )

        # Sessões com modo /debug ativado (rastreado pelo ID do usuário)
        self.active_debug_users: Set[str] = set()

    def is_debug_active(self, user_id: int | str) -> bool:
        """Verifica se o usuário atual está com o modo /debug habilitado."""
        return str(user_id) in self.active_debug_users

    def toggle_debug(self, user_id: int | str) -> bool:
        """Alterna o status do modo /debug (liga/desliga) para um usuário."""
        uid = str(user_id)
        if uid in self.active_debug_users:
            self.active_debug_users.remove(uid)
            return False
        else:
            self.active_debug_users.add(uid)
            return True

    def format_debug_block(self, text: str, pred: PredictionResult) -> str:
        """Formata didaticamente o raio-X completo do pipeline de PLN da mensagem."""
        inspection = self.nlp.inspect(text)
        lr_candidates = pred.top_candidates
        top_terms = pred.active_tfidf_terms[:5]

        debug_text = [
            "🔬 **[MODO DEBUG — RAIO-X DO PIPELINE DE PLN]**",
            f"1️⃣ Texto Original: \"{inspection['1_texto_original']}\"",
            f"2️⃣ Normalização (minúsculas): \"{inspection['2_normalizado']}\"",
            f"3️⃣ Expansão de Gírias: \"{inspection['3_expansao_girias']}\"",
            f"4️⃣ Limpeza Regex (ruídos): \"{inspection['4_limpeza_regex']}\"",
            f"5️⃣ Tokens: {inspection['5_tokens_brutos']}",
            f"6️⃣ Sem Stopwords (negações mantidas): {inspection['6_apos_stopwords']}",
            f"7️⃣ Radicais (Stemming didático): {inspection['7_radicais_stemming']}",
            f"8️⃣ Termos de Maior Peso TF-IDF: {top_terms if top_terms else 'Nenhum termo do vocabulário'}",
            "--------------------------------------------------",
            f"🤖 Modelo Primário: **{pred.primary_model}**",
            f"🎯 Intenção Predita: **{pred.intent}**",
            f"📈 Certeza: **{pred.confidence * 100:.1f}%** (Limiar Fallback: {CONFIDENCE_THRESHOLD * 100:.1f}%)",
            f"🛡️ Fallback Acionado: **{'SIM' if pred.is_fallback else 'NÃO'}**",
            "📊 Top 3 Hipóteses (Regressão Logística):"
        ]
        for rank, (intent, prob) in enumerate(lr_candidates, start=1):
            debug_text.append(f"   {rank}. {intent} ({prob * 100:.1f}%)")

        debug_text.append(
            f"💡 Predição Comparativa (Naive Bayes): {pred.nb_intent} ({pred.nb_confidence * 100:.1f}%)"
        )
        debug_text.append("==================================================")
        return "\n".join(debug_text)

    def handle_message(
        self,
        user_id: int | str,
        message: str,
        channel: str = "telegram"
    ) -> BotResponse:
        """Processa comandos ou perguntas normais e retorna a resposta formatada."""
        text_clean = message.strip()
        uid_str = str(user_id)

        # ----------------------------------------------------------------------
        # COMANDO: /start
        # ----------------------------------------------------------------------
        if text_clean.startswith("/start"):
            welcome_msg = (
                "👋 **Olá! Bem-vindo ao Chatbot Tutor de PLN e Chatbots!**\n\n"
                "🏛️ **Instituição:** CETAM — Escola IBC (Manaus/AM)\n"
                "🎓 **Curso:** Técnico de Nível Médio em Inteligência Artificial\n"
                "👨‍🏫 **Docente:** Prof. Agnaldo Marcelo Pereira Fournier\n\n"
                "🤖 Sou um assistente educacional construído com **Machine Learning clássico supervisionado "
                "(TF-IDF + Regressão Logística / Naive Bayes)**, sem uso de LLMs.\n\n"
                "🔒 **Aviso de Privacidade (LGPD):** Esta conversa tem finalidade estritamente pedagógica. "
                "Seu ID e mensagens são anonimizados automaticamente antes de serem registrados no log.\n\n"
                "💡 **O que você pode fazer:**\n"
                "• Fazer perguntas normais em português sobre a matéria, notas e o curso;\n"
                "• Digitar `/ajuda` para ver todos os comandos disponíveis;\n"
                "• Digitar `/debug` para ver o raio-X matemático de como analiso cada frase!"
            )
            return BotResponse(text=welcome_msg, action_type="text")

        # ----------------------------------------------------------------------
        # COMANDO: /ajuda ou /help
        # ----------------------------------------------------------------------
        if text_clean.startswith(("/ajuda", "/help")):
            help_msg = (
                "📖 **GUIA DE COMANDOS DO TUTOR:**\n\n"
                "• `/start` - Apresentação da disciplina, curso e política de privacidade;\n"
                "• `/ajuda` - Este menu de instruções;\n"
                "• `/debug` - Liga/desliga o modo raio-X do pipeline de PLN para suas mensagens;\n"
                "• `/debug <frase>` - Inspeciona imediatamente os passos de PLN de uma frase avulsa;\n"
                "• `/metricas` - Exibe a acurácia, F1-Score e a matriz de confusão dos modelos;\n"
                "• `/regenerar` - (Apenas Professores/Admins) Curadoria de dúvidas para o bot aprender!\n\n"
                "💬 **Exemplos de perguntas que você pode me fazer:**\n"
                "• *'Qual é o nome do professor?'*\n"
                "• *'Como calcula a média das avaliações AV1 e AV2?'*\n"
                "• *'O que é TF-IDF?'*\n"
                "• *'Qual a diferença entre Stemming e Lematização?'*\n"
                "• *'Por que não podemos remover a palavra não nas stopwords?'*"
            )
            return BotResponse(text=help_msg, action_type="text")

        # ----------------------------------------------------------------------
        # COMANDO: /debug ou /debug <frase>
        # ----------------------------------------------------------------------
        if text_clean.startswith("/debug"):
            args = text_clean[6:].strip()
            if args:
                # Inspeciona uma frase específica sob demanda
                pred = self.ml.predict(args)
                debug_output = self.format_debug_block(args, pred)
                return BotResponse(text=debug_output, is_debug=True, action_type="text")
            else:
                # Alterna o estado da sessão do usuário
                new_state = self.toggle_debug(user_id)
                status_str = "🟢 **ATIVADO**" if new_state else "🔴 **DESATIVADO**"
                msg = (
                    f"Modo Debug {status_str} para a sua sessão!\n"
                    f"{'A partir de agora, cada resposta virá acompanhada do raio-X do pipeline de PLN.' if new_state else 'As próximas respostas voltarão ao modo normal de atendimento.'}"
                )
                return BotResponse(text=msg, is_debug=new_state, action_type="text")

        # ----------------------------------------------------------------------
        # COMANDO: /metricas
        # ----------------------------------------------------------------------
        if text_clean.startswith("/metricas"):
            report = self.evaluator.format_report_text()
            return BotResponse(text=report, action_type="text")

        # ----------------------------------------------------------------------
        # COMANDO: /regenerar (Controle de Acesso Administrativo)
        # ----------------------------------------------------------------------
        if text_clean.startswith("/regenerar"):
            # No canal CLI local o acesso é irrestrito (professor no console);
            # No Telegram, exige que o ID esteja na WHITELIST do .env
            if channel == "telegram" and not is_user_admin(user_id):
                refusal = (
                    "⛔ **Acesso Restrito:**\n"
                    "O comando `/regenerar` modifica o dataset de treinamento do robô e é restrito "
                    "aos professores e administradores cadastrados na `TELEGRAM_WHITELIST` do `.env`.\n"
                    "Peça ao professor para realizar a curadoria pelo terminal local!"
                )
                return BotResponse(text=refusal, action_type="unauthorized")

            pending = self.improver.get_pending_fallbacks()
            if not pending:
                msg = (
                    "🎉 **Nenhuma mensagem pendente de curadoria!**\n"
                    "Todas as dúvidas em que o robô caiu no Fallback já foram revisadas ou não há novos registros no log."
                )
                return BotResponse(text=msg, action_type="text")

            # Monta resumo das mensagens pendentes
            lines = [
                f"📋 **CURADORIA SUPERVISIONADA (/regenerar)**",
                f"Encontrei {len(pending)} mensagem(ns) onde o modelo teve baixa certeza:\n"
            ]
            for idx, item in enumerate(pending[:5], start=1):
                lines.append(f"{idx}. *\"{item['mensagem_anonimizada']}\"* (Data: {item['data']} {item['hora']})")

            lines.append(
                "\nPara associar uma frase ao dataset, use o terminal local ou informe a intenção desejada."
            )
            return BotResponse(text="\n".join(lines), action_type="regenerate_flow")

        # ----------------------------------------------------------------------
        # FLUXO PADRÃO: INFERÊNCIA DE PERGUNTAS DO USUÁRIO
        # ----------------------------------------------------------------------
        prediction = self.ml.predict(text_clean)

        # Registro persistente e anonimizado em conformidade com a LGPD
        self.logger.log_interaction(
            raw_user_id=user_id,
            raw_message=text_clean,
            predicted_intent=prediction.intent,
            confidence=prediction.confidence,
            is_fallback=prediction.is_fallback,
            channel=channel
        )

        # Se o modo debug estiver ativo, anexa o bloco técnico antes da resposta
        final_text = prediction.response
        if self.is_debug_active(user_id):
            debug_block = self.format_debug_block(text_clean, prediction)
            final_text = f"{debug_block}\n\n💬 **Resposta do Tutor:**\n{prediction.response}"

        return BotResponse(
            text=final_text,
            is_debug=self.is_debug_active(user_id),
            prediction=prediction,
            action_type="text"
        )
