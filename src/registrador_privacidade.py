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
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.configuracao import (
    ARQUIVO_LOG_CSV,
    ARQUIVO_LOG_JSONL,
    DIRETORIO_LOGS,
    SAL_ANONIMIZACAO,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# PADRÕES DE EXPRESSÕES REGULARES PARA DADOS SENSÍVEIS (PII)
# ==============================================================================
# Padrão de CPF (com ou sem pontuação)
REGEX_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}[-.]?\d{2}\b")

# Padrão de E-mail
REGEX_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Padrão de Telefone celular / fixo no Brasil (com ou sem DDD e traços)
REGEX_TELEFONE = re.compile(
    r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?[\s-]?)?(?:9\s?\d{4}|\d{4})[-\s]?\d{4}\b"
)

# Padrão de Matrícula estudantil ou RA
REGEX_MATRICULA = re.compile(r"\b(?:matr[ií]cula|ra|id)\s*[:=]?\s*\d{4,}\b", re.IGNORECASE)

# Padrão de declaração de nomes próprios em português:
# Nota de Segurança: O gatilho ('meu nome é', 'me chamo', 'sou o/a') é insensível a maiúsculas
# via (?i:...), mas o nome próprio capturado exige letra inicial maiúscula ([A-ZÁÉÍÓÚ...]),
# evitando que frases como "sou o aluno que faltou ontem" tenham seu conteúdo apagado!
REGEX_NOME_DECLARADO = re.compile(
    r"\b((?i:meu nome [ée]|me chamo|sou [oa]))\s+"
    r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:(?:da|de|do|das|dos)\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*)\b"
)

CABECALHOS_CSV: List[str] = [
    "id_interacao",
    "data",
    "hora",
    "canal",
    "usuario_anonimizado",
    "mensagem_anonimizada",
    "intencao_identificada",
    "probabilidade_confianca",
    "fallback_acionado",
    "revisado_aprendizado",
]


class RegistradorPrivacidade:
    """Gerenciador de logs auditáveis e anonimizados de conversas."""

    def __init__(
        self,
        caminho_jsonl: Path | None = None,
        caminho_csv: Path | None = None,
        sal: str = SAL_ANONIMIZACAO
    ) -> None:
        self.caminho_jsonl: Path = caminho_jsonl or ARQUIVO_LOG_JSONL
        self.caminho_csv: Path = caminho_csv or ARQUIVO_LOG_CSV
        self.sal: str = sal

        DIRETORIO_LOGS.mkdir(parents=True, exist_ok=True)
        self._garantir_cabecalho_csv()

    def _garantir_cabecalho_csv(self) -> None:
        """Garante a criação do cabeçalho caso o arquivo CSV não exista."""
        if not self.caminho_csv.exists():
            try:
                self.caminho_csv.parent.mkdir(parents=True, exist_ok=True)
                with open(self.caminho_csv, mode="w", encoding="utf-8", newline="") as f:
                    escritor = csv.writer(f)
                    escritor.writerow(CABECALHOS_CSV)
            except Exception as erro:
                logger.warning(f"Aviso: Falha ao criar cabeçalho CSV em '{self.caminho_csv}': {erro}")

    def anonimizar_id_usuario(self, id_usuario_bruto: int | str) -> str:
        """Gera um identificador pseudo-anônimo determinístico com hash SHA-256 e Sal.

        Permite correlacionar mensagens da mesma sessão sem jamais revelar o ID real do Telegram.
        """
        combinado = f"{self.sal}_{id_usuario_bruto}".encode("utf-8")
        resumo_hash = hashlib.sha256(combinado).hexdigest()
        return f"usr_{resumo_hash[:16]}"

    def sanitizar_mensagem(self, mensagem: str) -> str:
        """Substitui dados pessoais identificáveis por marcadores genéricos de privacidade."""
        sanitizada = mensagem

        # 1. Oculta CPF
        sanitizada = REGEX_CPF.sub("[CPF_OCULTO]", sanitizada)

        # 2. Oculta E-mails
        sanitizada = REGEX_EMAIL.sub("[EMAIL_OCULTO]", sanitizada)

        # 3. Oculta Telefones
        sanitizada = REGEX_TELEFONE.sub("[TELEFONE_OCULTO]", sanitizada)

        # 4. Oculta Matrícula estudantil
        sanitizada = REGEX_MATRICULA.sub("[MATRICULA_OCULTA]", sanitizada)

        # 5. Oculta nomes próprios capitalizados após apresentações
        sanitizada = REGEX_NOME_DECLARADO.sub(r"\1 [NOME_OCULTO]", sanitizada)

        return sanitizada

    def registrar_interacao(
        self,
        id_usuario_bruto: int | str,
        mensagem_bruta: str,
        intencao_predita: str,
        confianca: float,
        acionou_fallback: bool,
        canal: str = "telegram"
    ) -> Dict[str, Any]:
        """Anonimiza e persiste o registro da interação em formato JSONL e CSV."""
        agora = datetime.now()
        id_registro = str(uuid.uuid4())
        usuario_anonimo = self.anonimizar_id_usuario(id_usuario_bruto)
        mensagem_anonima = self.sanitizar_mensagem(mensagem_bruta)

        entrada_log: Dict[str, Any] = {
            "id_interacao": id_registro,
            "data": agora.strftime("%Y-%m-%d"),
            "hora": agora.strftime("%H:%M:%S"),
            "canal": canal,
            "usuario_anonimizado": usuario_anonimo,
            "mensagem_anonimizada": mensagem_anonima,
            "intencao_identificada": intencao_predita,
            "probabilidade_confianca": round(confianca, 4),
            "fallback_acionado": acionou_fallback,
            "revisado_aprendizado": False
        }

        # 1. Gravação no formato JSONL (Acréscimo atômico linha por linha)
        try:
            with open(self.caminho_jsonl, mode="a", encoding="utf-8") as f_json:
                f_json.write(json.dumps(entrada_log, ensure_ascii=False) + "\n")
                f_json.flush()
        except Exception as erro:
            logger.warning(f"Aviso: Falha ao gravar log JSONL em '{self.caminho_jsonl}': {erro}")

        # 2. Gravação no formato CSV
        try:
            self._garantir_cabecalho_csv()
            with open(self.caminho_csv, mode="a", encoding="utf-8", newline="") as f_csv:
                escritor = csv.writer(f_csv)
                escritor.writerow([entrada_log[col] for col in CABECALHOS_CSV])
                f_csv.flush()
        except Exception as erro:
            logger.warning(f"Aviso: Falha ao gravar log CSV em '{self.caminho_csv}': {erro}")

        return entrada_log

    def ler_todos_logs(self) -> List[Dict[str, Any]]:
        """Lê todos os registros de log armazenados no arquivo JSONL."""
        registros: List[Dict[str, Any]] = []
        if not self.caminho_jsonl.exists():
            return registros

        try:
            with open(self.caminho_jsonl, mode="r", encoding="utf-8") as f:
                for linha in f:
                    linha_limpa = linha.strip()
                    if linha_limpa:
                        try:
                            registros.append(json.loads(linha_limpa))
                        except json.JSONDecodeError:
                            continue
        except Exception as erro:
            logger.warning(f"Aviso: Falha ao ler logs em '{self.caminho_jsonl}': {erro}")

        return registros

    def marcar_interacao_revisada(self, id_interacao: str) -> bool:
        """Marca uma interação como revisada tanto no arquivo JSONL quanto no CSV."""
        if not self.caminho_jsonl.exists():
            return False

        logs_atualizados: List[Dict[str, Any]] = []
        encontrado = False

        try:
            with open(self.caminho_jsonl, mode="r", encoding="utf-8") as f:
                for linha in f:
                    linha_limpa = linha.strip()
                    if not linha_limpa:
                        continue
                    try:
                        dados = json.loads(linha_limpa)
                        if dados.get("id_interacao") == id_interacao:
                            dados["revisado_aprendizado"] = True
                            encontrado = True
                        logs_atualizados.append(dados)
                    except json.JSONDecodeError:
                        continue

            if encontrado:
                # 1. Atualiza JSONL
                with open(self.caminho_jsonl, mode="w", encoding="utf-8") as f_json:
                    for item in logs_atualizados:
                        f_json.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f_json.flush()

                # 2. Sincroniza CSV com os mesmos registros atualizados
                with open(self.caminho_csv, mode="w", encoding="utf-8", newline="") as f_csv:
                    escritor = csv.writer(f_csv)
                    escritor.writerow(CABECALHOS_CSV)
                    for item in logs_atualizados:
                        escritor.writerow([item.get(col, "") for col in CABECALHOS_CSV])
                    f_csv.flush()
        except Exception as erro:
            logger.warning(f"Aviso: Falha ao atualizar status de revisão do log '{id_interacao}': {erro}")
            return False

        return encontrado

    def obter_resumo_metricas(self) -> Dict[str, Any]:
        """Calcula métricas operacionais a partir do histórico de conversas."""
        logs = self.ler_todos_logs()
        total_interacoes = len(logs)
        if total_interacoes == 0:
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
        resolvidas = total_interacoes - fallbacks
        confiancas = [float(item.get("probabilidade_confianca", 0.0)) for item in logs]
        usuarios_unicos = len({item.get("usuario_anonimizado", "") for item in logs})

        # Contagem de intenções
        contagem_intencoes: Dict[str, int] = {}
        for item in logs:
            if not item.get("fallback_acionado", False):
                intencao = item.get("intencao_identificada", "")
                if intencao:
                    contagem_intencoes[intencao] = contagem_intencoes.get(intencao, 0) + 1

        top_intencoes = sorted(contagem_intencoes.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_interacoes": total_interacoes,
            "total_resolvidas": resolvidas,
            "total_fallback": fallbacks,
            "taxa_resolucao": round((resolvidas / total_interacoes) * 100, 2),
            "taxa_fallback": round((fallbacks / total_interacoes) * 100, 2),
            "confianca_media": round(float(sum(confiancas) / total_interacoes), 3),
            "usuarios_unicos": usuarios_unicos,
            "top_intencoes": top_intencoes
        }
