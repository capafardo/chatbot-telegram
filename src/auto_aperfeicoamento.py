"""Módulo de Auto-Aperfeiçoamento Supervisionado (/regenerar).

Referência Didática:
    - Requisitos 7 e 15 da especificação e Aula 07/08 do Plano de Ensino.
    - Curadoria humana e aprendizado contínuo supervisionado.
    - O chatbot identifica dúvidas em que caiu no Fallback, permitindo que o professor
      ou administrador vincule a nova frase a uma intenção existente ou nova,
      criando backup automático e retreinando o modelo em memória imediatamente.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.configuracao import BACKUP_DIR, INTENTS_FILE, LOG_JSONL_FILE
from src.motor_ml import MLEngine
from src.registrador_privacidade import PrivacyLogger


class SelfImprover:
    """Orquestrador de curadoria de mensagens não compreendidas e retreinamento."""

    def __init__(
        self,
        ml_engine: MLEngine | None = None,
        privacy_logger: PrivacyLogger | None = None,
        intents_path: Path | None = None,
        backup_dir: Path | None = None
    ) -> None:
        self.engine: MLEngine = ml_engine or MLEngine()
        self.logger: PrivacyLogger = privacy_logger or PrivacyLogger()
        self.intents_path: Path = intents_path or INTENTS_FILE
        self.backup_dir: Path = backup_dir or BACKUP_DIR

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> Path:
        """Gera uma cópia de segurança com carimbo de data/hora do dataset intents.json."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"intents_backup_{timestamp}.json"
        shutil.copy2(self.intents_path, backup_file)
        return backup_file

    def get_pending_fallbacks(self) -> List[Dict[str, Any]]:
        """Recupera mensagens do log que caíram em Fallback e ainda não foram revisadas.

        Aplica filtros de qualidade para evitar ruídos de digitação puros ou emojis avulsos.
        """
        logs = self.logger.read_all_logs()
        candidates: List[Dict[str, Any]] = []
        seen_texts: set[str] = set()

        for entry in logs:
            if entry.get("fallback_acionado", False) and not entry.get("revisado_aprendizado", False):
                raw_msg = entry.get("mensagem_anonimizada", "").strip()
                # Validação de qualidade: mínimo de 4 caracteres alfabéticos
                alpha_chars = re.findall(r"[A-Za-zÀ-ÿ]", raw_msg)
                if len(alpha_chars) >= 4 and raw_msg.lower() not in seen_texts:
                    seen_texts.add(raw_msg.lower())
                    candidates.append(entry)

        return candidates

    def mark_log_reviewed(self, interaction_id: str) -> bool:
        """Atualiza o registro no arquivo JSONL para indicar que a mensagem foi revisada."""
        if not LOG_JSONL_FILE.exists():
            return False

        updated_logs: List[Dict[str, Any]] = []
        found = False

        with open(LOG_JSONL_FILE, mode="r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if data.get("id_interacao") == interaction_id:
                        data["revisado_aprendizado"] = True
                        found = True
                    updated_logs.append(data)
                except json.JSONDecodeError:
                    continue

        if found:
            with open(LOG_JSONL_FILE, mode="w", encoding="utf-8") as f:
                for item in updated_logs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

        return found

    def add_phrase_to_intent(
        self,
        intent_tag: str,
        phrase: str,
        interaction_id: str | None = None
    ) -> tuple[bool, str]:
        """Adiciona uma nova frase de treino a uma intenção existente e retreina o modelo."""
        clean_phrase = phrase.strip()
        if not clean_phrase:
            return False, "Frase vazia não pode ser adicionada ao dataset."

        # 1. Criação do backup de segurança
        try:
            self.create_backup()
        except Exception as e:
            return False, f"Falha ao criar backup do dataset: {e}"

        # 2. Carrega dataset JSON atual
        try:
            with open(self.intents_path, mode="r", encoding="utf-8") as f:
                dataset = json.load(f)
        except Exception as e:
            return False, f"Erro ao ler intents.json: {e}"

        # 3. Localiza a intenção e insere a nova frase
        intent_found = False
        for intent in dataset.get("intents", []):
            if intent["tag"] == intent_tag:
                intent_found = True
                patterns = intent.setdefault("patterns", [])
                # Evita duplicatas case-insensitive
                if not any(p.lower() == clean_phrase.lower() for p in patterns):
                    patterns.append(clean_phrase)
                break

        if not intent_found:
            return False, f"Intenção '{intent_tag}' não encontrada no dataset."

        # 4. Grava dataset atualizado
        try:
            with open(self.intents_path, mode="w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
                f.flush()
        except Exception as e:
            return False, f"Erro ao salvar intents.json atualizado: {e}"

        # 5. Marca mensagem no log como revisada
        if interaction_id:
            self.mark_log_reviewed(interaction_id)

        # 6. Retreinamento em tempo de execução (Hot Retrain)
        try:
            self.engine.load_and_train()
        except Exception as e:
            return False, f"Frase salva, mas erro ao retreinar modelo: {e}"

        return True, f"Frase adicionada com sucesso à intenção '{intent_tag}'. Modelo retreinado!"

    def create_new_intent(
        self,
        tag: str,
        categoria: str,
        descricao: str,
        first_pattern: str,
        first_response: str,
        interaction_id: str | None = None
    ) -> tuple[bool, str]:
        """Cria uma nova categoria de intenção no dataset e retreina o classificador."""
        tag_clean = tag.strip().lower().replace(" ", "_")
        if not tag_clean:
            return False, "A tag da intenção não pode ser vazia."

        # Backup
        self.create_backup()

        with open(self.intents_path, mode="r", encoding="utf-8") as f:
            dataset = json.load(f)

        # Verifica se já existe
        for intent in dataset.get("intents", []):
            if intent["tag"] == tag_clean:
                return False, f"A intenção '{tag_clean}' já existe no dataset."

        new_intent_block = {
            "tag": tag_clean,
            "categoria": categoria or "tecnica",
            "descricao": descricao or f"Intenção criada dinamicamente: {tag_clean}",
            "patterns": [first_pattern.strip()],
            "responses": [first_response.strip()]
        }

        dataset["intents"].append(new_intent_block)

        with open(self.intents_path, mode="w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            f.flush()

        if interaction_id:
            self.mark_log_reviewed(interaction_id)

        self.engine.load_and_train()
        return True, f"Nova intenção '{tag_clean}' criada com sucesso. Modelo retreinado!"
