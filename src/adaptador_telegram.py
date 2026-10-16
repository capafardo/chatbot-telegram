"""Adaptador de Integração com o Telegram Bot API (Modo Polling).

Referência Didática:
    - Aula 08 do Plano de Ensino (CETAM / IBC).
    - Integração de assistentes virtuais via API de mensageria usando método Polling.
    - Segurança: uso de variáveis de ambiente (.env) para proteção do token e Whitelist.
"""

from __future__ import annotations

import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.controlador_bot import BotController
from src.configuracao import TELEGRAM_API_KEY, validate_telegram_config

# Configuração de logging informativo
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramAdapter:
    """Adaptador de mensageria assíncrona para o Telegram Bot API."""

    def __init__(self, controller: BotController | None = None) -> None:
        self.controller: BotController = controller or BotController()
        self.application: Optional[Application] = None

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /start."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        response = self.controller.handle_message(user_id, "/start", channel="telegram")
        await update.message.reply_text(response.text)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /ajuda."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        response = self.controller.handle_message(user_id, "/ajuda", channel="telegram")
        await update.message.reply_text(response.text)

    async def _handle_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /debug."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        msg_text = update.message.text or "/debug"
        response = self.controller.handle_message(user_id, msg_text, channel="telegram")
        await update.message.reply_text(response.text)

    async def _handle_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /metricas."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        response = self.controller.handle_message(user_id, "/metricas", channel="telegram")
        await update.message.reply_text(response.text)

    async def _handle_regenerate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador para o comando /regenerar."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        response = self.controller.handle_message(user_id, "/regenerar", channel="telegram")
        await update.message.reply_text(response.text)

    async def _handle_incoming_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manipulador de mensagens de texto regulares enviadas pelos estudantes."""
        if not update.effective_user or not update.message or not update.message.text:
            return

        user_id = update.effective_user.id
        raw_text = update.message.text

        response = self.controller.handle_message(user_id, raw_text, channel="telegram")
        await update.message.reply_text(response.text)

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Tratamento global de exceções na conexão com o Telegram."""
        logger.error(f"Exceção ocorrida ao processar atualização: {context.error}")

    def build_application(self) -> Application:
        """Constrói e registra os manipuladores de eventos do bot."""
        valid, msg = validate_telegram_config()
        if not valid:
            raise ValueError(msg)

        app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()

        # Registro de comandos
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("ajuda", self._handle_help))
        app.add_handler(CommandHandler("help", self._handle_help))
        app.add_handler(CommandHandler("debug", self._handle_debug))
        app.add_handler(CommandHandler("metricas", self._handle_metrics))
        app.add_handler(CommandHandler("regenerar", self._handle_regenerate))

        # Mensagens normais de texto
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_incoming_text))

        # Tratamento de erro
        app.add_error_handler(self._handle_error)

        self.application = app
        return app

    def start_polling(self) -> None:
        """Inicia a escuta contínua de mensagens pelo método Polling."""
        print("Iniciando conexão com a API do Telegram (Modo Polling)...")
        app = self.build_application()
        print("Bot conectado com sucesso! Pressione Ctrl+C para encerrar.")
        app.run_polling()
