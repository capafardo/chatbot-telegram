"""Adaptador de Integração com o Telegram Bot API (Modo Polling).

Referência Didática:
    - Aula 08 do Plano de Ensino (CETAM / IBC).
    - Integração de assistentes virtuais via API de mensageria usando método Polling.
    - Segurança: uso de variáveis de ambiente (.env) para proteção do token e Whitelist.
    - Formatação HTML segura (evitando falhas com underscores em tags) e divisão automática
      de mensagens longas (> 4096 caracteres).
"""

from __future__ import annotations

import html
import logging
import re
from typing import List, Optional

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.configuracao import CHAVE_API_TELEGRAM, NIVEL_LOG, validar_config_telegram
from src.controlador_bot import ControladorBot

# Configuração de logging informativo
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, NIVEL_LOG, logging.INFO)
)
logger = logging.getLogger(__name__)

LIMITE_CARACTERES_TELEGRAM: int = 4000


def formatar_para_html_telegram(texto: str) -> str:
    """Converte marcações Markdown simples (**negrito**, *itálico*, `código`) para HTML seguro do Telegram.

    Vantagem didática e técnica:
        O modo Markdown padrão do Telegram quebra quando encontra caracteres '_' não escapados
        (muito comuns em nomes de intenções como `conceito_pln_historico`). Ao escapar o HTML
        primeiro e converter para `<b>`, `<i>` e `<code>`, garantimos formatação visual rica
        sem risco de exceção `BadRequest: Can't parse entities`.
    """
    escapado = html.escape(texto)
    # 1. Blocos inline de código: `codigo` -> <code>codigo</code>
    escapado = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escapado)
    # 2. Negrito: **texto** -> <b>texto</b>
    escapado = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escapado)
    # 3. Itálico: *texto* -> <i>texto</i>
    escapado = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escapado)
    return escapado


def dividir_mensagem_longa(texto: str, limite: int = LIMITE_CARACTERES_TELEGRAM) -> List[str]:
    """Divide textos longos respeitando quebras de linha e o limite de 4096 caracteres do Telegram."""
    if len(texto) <= limite:
        return [texto]

    partes: List[str] = []
    bloco_atual: List[str] = []
    tamanho_atual = 0

    for linha in texto.split("\n"):
        tamanho_linha = len(linha) + 1
        if tamanho_atual + tamanho_linha > limite and bloco_atual:
            partes.append("\n".join(bloco_atual))
            bloco_atual = [linha]
            tamanho_atual = tamanho_linha
        else:
            bloco_atual.append(linha)
            tamanho_atual += tamanho_linha

    if bloco_atual:
        partes.append("\n".join(bloco_atual))

    return partes


class AdaptadorTelegram:
    """Adaptador de mensageria assíncrona para o Telegram Bot API."""

    def __init__(self, controlador: ControladorBot | None = None) -> None:
        self.controlador: ControladorBot = controlador or ControladorBot()
        self.aplicacao: Optional[Application] = None

    async def _enviar_resposta_segura(self, update: Update, texto: str) -> None:
        """Envia a resposta formatada em HTML (e fatiada se necessário) com fallback para texto puro."""
        if not update.message:
            return

        partes = dividir_mensagem_longa(texto)
        for parte in partes:
            texto_html = formatar_para_html_telegram(parte)
            try:
                await update.message.reply_text(texto_html, parse_mode="HTML")
            except Exception as erro:
                logger.warning(f"Fallback para texto puro ao enviar mensagem no Telegram: {erro}")
                await update.message.reply_text(parte)

    async def _tratar_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /start."""
        if not update.effective_user or not update.message:
            return

        id_usuario = update.effective_user.id
        resposta = self.controlador.processar_mensagem(id_usuario, "/start", canal="telegram")
        await self._enviar_resposta_segura(update, resposta.texto)

    async def _tratar_ajuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /ajuda."""
        if not update.effective_user or not update.message:
            return

        id_usuario = update.effective_user.id
        resposta = self.controlador.processar_mensagem(id_usuario, "/ajuda", canal="telegram")
        await self._enviar_resposta_segura(update, resposta.texto)

    async def _tratar_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /debug."""
        if not update.effective_user or not update.message:
            return

        id_usuario = update.effective_user.id
        texto_msg = update.message.text or "/debug"
        resposta = self.controlador.processar_mensagem(id_usuario, texto_msg, canal="telegram")
        await self._enviar_resposta_segura(update, resposta.texto)

    async def _tratar_metricas(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /metricas."""
        if not update.effective_user or not update.message:
            return

        id_usuario = update.effective_user.id
        resposta = self.controlador.processar_mensagem(id_usuario, "/metricas", canal="telegram")
        await self._enviar_resposta_segura(update, resposta.texto)

    async def _tratar_regenerar(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /regenerar."""
        if not update.effective_user or not update.message:
            return

        id_usuario = update.effective_user.id
        resposta = self.controlador.processar_mensagem(id_usuario, "/regenerar", canal="telegram")
        await self._enviar_resposta_segura(update, resposta.texto)

    async def _tratar_texto_recebido(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador de mensagens de texto regulares enviadas pelos estudantes."""
        if not update.effective_user or not update.message or not update.message.text:
            return

        id_usuario = update.effective_user.id
        texto_bruto = update.message.text

        resposta = self.controlador.processar_mensagem(id_usuario, texto_bruto, canal="telegram")
        await self._enviar_resposta_segura(update, resposta.texto)

    async def _tratar_erro(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Tratamento global de exceções na conexão com o Telegram."""
        logger.error(f"Exceção ocorrida ao processar atualização: {context.error}")

    async def _pos_inicializacao(self, aplicacao: Application) -> None:
        """Registra automaticamente o menu de comandos na interface do Telegram ao conectar."""
        comandos = [
            BotCommand("start", "Apresentação e aviso de privacidade (LGPD)"),
            BotCommand("ajuda", "Guia de comandos e exemplos de perguntas"),
            BotCommand("debug", "Liga/desliga o raio-X do pipeline de PLN"),
            BotCommand("metricas", "Exibe acurácia, F1-Score e matriz de confusão"),
            BotCommand("regenerar", "(Admin) Painel de curadoria de dúvidas"),
        ]
        try:
            await aplicacao.bot.set_my_commands(comandos)
        except Exception as erro:
            logger.warning(f"Não foi possível registrar o menu de comandos no Telegram: {erro}")

    def construir_aplicacao(self) -> Application:
        """Constrói e registra os manipuladores de eventos do bot."""
        valido, msg = validar_config_telegram()
        if not valido:
            raise ValueError(msg)

        app = (
            ApplicationBuilder()
            .token(CHAVE_API_TELEGRAM)
            .post_init(self._pos_inicializacao)
            .build()
        )

        # Registro de comandos
        app.add_handler(CommandHandler("start", self._tratar_start))
        app.add_handler(CommandHandler("ajuda", self._tratar_ajuda))
        app.add_handler(CommandHandler("help", self._tratar_ajuda))
        app.add_handler(CommandHandler("debug", self._tratar_debug))
        app.add_handler(CommandHandler("metricas", self._tratar_metricas))
        app.add_handler(CommandHandler("regenerar", self._tratar_regenerar))

        # Mensagens normais de texto
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._tratar_texto_recebido)
        )

        # Tratamento de erro
        app.add_error_handler(self._tratar_erro)

        self.aplicacao = app
        return app

    def iniciar_polling(self) -> None:
        """Inicia a escuta contínua de mensagens pelo método Polling."""
        ident = self.controlador.obter_identidade_bot()
        print(f"Iniciando '{ident['nome_bot']}' na API do Telegram (Modo Polling)...")
        app = self.construir_aplicacao()
        print("Bot conectado com sucesso! Pressione Ctrl+C para encerrar.")
        app.run_polling()
