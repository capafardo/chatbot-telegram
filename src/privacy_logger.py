"""Módulo de Anonimização Ética e Persistência de Logs em Conformidade com a LGPD.

Referência Didática:
    - Aula 09 do Plano de Ensino (CETAM / IBC).
    - Princípios da LGPD: Finalidade, Adequação, Necessidade e Segurança.
    - Transformação irreversível de identificadores de usuários via hash criptográfico SHA-256 com salt.
    - Ocultação de dados pessoais identificáveis (PII) por Expressões Regulares antes da gravação em disco.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import LOG_CSV_FILE, LOG_JSONL_FILE, LOGS_DIR, SALT_ANONYMIZATION

# ==============================================================================
# PADRÕES DE EXPRESSÕES REGULARES PARA DADOS SENSÍVEIS (PII)
# ==============================================================================
# Padrão de CPF (com ou sem pontuação)
REGEX_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}[-.]?\d{2}\b")

# Padrão de E-mail
REGEX_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Padrão de Telefone celular / fixo no Brasil (com ou sem DDD e traços)
REGEX_PHONE = re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?[\s-]?)?(?:9\s?\d{4}|\d{4})[-\s]?\d{4}\b")

# Padrão de Matrícula estudantil ou RA
REGEX_MATRICULA = re.compile(r"\b(?:matr[ií]cula|ra|id)\s*[:=]?\s*\d{4,}\b", re.IGNORECASE)

# Padrão de declaração de nomes próprios em português
REGEX_NOME_DECLARADO = re.compile(
    r"\b(meu nome [ée]|me chamo|sou [oa])\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*)\b",
    re.IGNORECASE
)


class PrivacyLogger:
    """Gerenciador de logs auditáveis e anonimizados de conversas."""

    def __init__(
        self,
        jsonl_path: Path | None = None,
        csv_path: Path | None = None,
        salt: str = SALT_ANONYMIZATION
    ) -> None:
        self.jsonl_path: Path = jsonl_path or LOG_JSONL_FILE
        self.csv_path: Path = csv_path or LOG_CSV_FILE
        self.salt: str = salt

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Garante a criação do cabeçalho caso o arquivo CSV não exista."""
        if not self.csv_path.exists():
            headers = [
                "id_interacao",
                "data",
                "hora",
                "canal",
                "usuario_anonimizado",
                "mensagem_anonimizada",
                "intencao_identificada",
                "probabilidade_confianca",
                "fallback_acionado",
                "revisado_aprendizado"
            ]
            try:
                with open(self.csv_path, mode="w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            except Exception:
                pass

    def anonymize_user_id(self, raw_user_id: int | str) -> str:
        """Gera um identificador pseudo-anônimo determinístico com hash SHA-256 e Salt.

        Permite correlacionar mensagens da mesma sessão sem jamais revelar o ID real do Telegram.
        """
        combined = f"{self.salt}_{raw_user_id}".encode("utf-8")
        hash_digest = hashlib.sha256(combined).hexdigest()
        return f"usr_{hash_digest[:16]}"

    def sanitize_message(self, message: str) -> str:
        """Substitui dados pessoais identificáveis por marcadores genéricos de privacidade."""
        sanitized = message

        # 1. Oculta CPF
        sanitized = REGEX_CPF.sub("[CPF_OCULTO]", sanitized)

        # 2. Oculta E-mails
        sanitized = REGEX_EMAIL.sub("[EMAIL_OCULTO]", sanitized)

        # 3. Oculta Telefones
        sanitized = REGEX_PHONE.sub("[TELEFONE_OCULTO]", sanitized)

        # 4. Oculta Matrícula estudantil
        sanitized = REGEX_MATRICULA.sub("[MATRICULA_OCULTA]", sanitized)

        # 5. Oculta nomes próprios após apresentações
        sanitized = REGEX_NOME_DECLARADO.sub(r"\1 [NOME_OCULTO]", sanitized)

        return sanitized

    def log_interaction(
        self,
        raw_user_id: int | str,
        raw_message: str,
        predicted_intent: str,
        confidence: float,
        is_fallback: bool,
        channel: str = "telegram"
    ) -> Dict[str, Any]:
        """Anonimiza e persiste o registro da interação em formato JSONL e CSV."""
        now = datetime.now()
        record_id = str(uuid.uuid4())
        user_anon = self.anonymize_user_id(raw_user_id)
        msg_anon = self.sanitize_message(raw_message)

        log_entry: Dict[str, Any] = {
            "id_interacao": record_id,
            "data": now.strftime("%Y-%m-%d"),
            "hora": now.strftime("%H:%M:%S"),
            "canal": channel,
            "usuario_anonimizado": user_anon,
            "mensagem_anonimizada": msg_anon,
            "intencao_identificada": predicted_intent,
            "probabilidade_confianca": round(confidence, 4),
            "fallback_acionado": is_fallback,
            "revisado_aprendizado": False
        }

        # 1. Gravação no formato JSONL (Acréscimo atômico linha por linha)
        try:
            with open(self.jsonl_path, mode="a", encoding="utf-8") as f_json:
                f_json.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f_json.flush()
        except Exception:
            pass

        # 2. Gravação no formato CSV
        try:
            with open(self.csv_path, mode="a", encoding="utf-8", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([
                    log_entry["id_interacao"],
                    log_entry["data"],
                    log_entry["hora"],
                    log_entry["canal"],
                    log_entry["usuario_anonimizado"],
                    log_entry["mensagem_anonimizada"],
                    log_entry["intencao_identificada"],
                    log_entry["probabilidade_confianca"],
                    log_entry["fallback_acionado"],
                    log_entry["revisado_aprendizado"]
                ])
                f_csv.flush()
        except Exception:
            pass

        return log_entry

    def read_all_logs(self) -> List[Dict[str, Any]]:
        """Lê todos os registros de log armazenados."""
        records: List[Dict[str, Any]] = []
        if not self.jsonl_path.exists():
            return records

        with open(self.jsonl_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        records.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calcula métricas operacionais a partir do histórico de conversas."""
        logs = self.read_all_logs()
        total_interactions = len(logs)
        if total_interactions == 0:
            return {
                "total_interacoes": 0,
                "total_resolvidas": 0,
                "total_fallback": 0,
                "taxa_resolucao": 0.0,
                "taxa_fallback": 0.0,
                "confianca_media": 0.0,
                "usuarios_unicos": 0,
                "top_intencoes": []
            }

        fallbacks = sum(1 for item in logs if item.get("fallback_acionado", False))
        resolved = total_interactions - fallbacks
        confidences = [item.get("probabilidade_confianca", 0.0) for item in logs]
        unique_users = len({item.get("usuario_anonimizado", "") for item in logs})

        # Contagem de intenções
        intent_counts: Dict[str, int] = {}
        for item in logs:
            if not item.get("fallback_acionado", False):
                intent = item.get("intencao_identificada", "")
                if intent:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1

        top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_interacoes": total_interactions,
            "total_resolvidas": resolved,
            "total_fallback": fallbacks,
            "taxa_resolucao": round((resolved / total_interactions) * 100, 2),
            "taxa_fallback": round((fallbacks / total_interactions) * 100, 2),
            "confianca_media": round(float(sum(confidences) / total_interactions), 3),
            "usuarios_unicos": unique_users,
            "top_intencoes": top_intents
        }
