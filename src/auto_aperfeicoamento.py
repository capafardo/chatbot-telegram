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
from typing import Any, Dict, List

from src.configuracao import ARQUIVO_INTENCOES, DIRETORIO_BACKUP
from src.motor_ml import MotorML
from src.registrador_privacidade import RegistradorPrivacidade


class AutoAperfeicoamento:
    """Orquestrador de curadoria de mensagens não compreendidas e retreinamento."""

    def __init__(
        self,
        motor_ml: MotorML | None = None,
        registrador_privacidade: RegistradorPrivacidade | None = None,
        caminho_intencoes: Path | None = None,
        diretorio_backup: Path | None = None
    ) -> None:
        self.motor: MotorML = motor_ml or MotorML()
        self.registrador: RegistradorPrivacidade = registrador_privacidade or RegistradorPrivacidade()
        self.caminho_intencoes: Path = caminho_intencoes or ARQUIVO_INTENCOES
        self.diretorio_backup: Path = diretorio_backup or DIRETORIO_BACKUP

        self.diretorio_backup.mkdir(parents=True, exist_ok=True)

    def criar_backup(self) -> Path:
        """Gera uma cópia de segurança com carimbo de data/hora do dataset intencoes.json."""
        carimbo_tempo = datetime.now().strftime("%Y%m%d_%H%M%S")
        arquivo_backup = self.diretorio_backup / f"intencoes_backup_{carimbo_tempo}.json"
        shutil.copy2(self.caminho_intencoes, arquivo_backup)
        return arquivo_backup

    def obter_fallbacks_pendentes(self) -> List[Dict[str, Any]]:
        """Recupera mensagens do log que caíram em Fallback e ainda não foram revisadas.

        Aplica filtros de qualidade para evitar ruídos de digitação puros ou emojis avulsos.
        """
        logs = self.registrador.ler_todos_logs()
        candidatos: List[Dict[str, Any]] = []
        textos_vistos: set[str] = set()

        for entrada in logs:
            if entrada.get("fallback_acionado", False) and not entrada.get("revisado_aprendizado", False):
                mensagem_bruta = entrada.get("mensagem_anonimizada", "").strip()
                # Validação de qualidade: mínimo de 4 caracteres alfabéticos
                caracteres_alfabeticos = re.findall(r"[A-Za-zÀ-ÿ]", mensagem_bruta)
                if len(caracteres_alfabeticos) >= 4 and mensagem_bruta.lower() not in textos_vistos:
                    textos_vistos.add(mensagem_bruta.lower())
                    candidatos.append(entrada)

        return candidatos

    def marcar_log_revisado(self, id_interacao: str) -> bool:
        """Marca a mensagem como revisada nos arquivos JSONL e CSV por meio do registrador."""
        return self.registrador.marcar_interacao_revisada(id_interacao)

    def adicionar_frase_a_intencao(
        self,
        tag_intencao: str,
        frase: str,
        id_interacao: str | None = None
    ) -> tuple[bool, str]:
        """Adiciona uma nova frase de treino a uma intenção existente e retreina o modelo."""
        frase_limpa = frase.strip()
        if not frase_limpa:
            return False, "Frase vazia não pode ser adicionada ao dataset."

        # 1. Criação do backup de segurança
        try:
            self.criar_backup()
        except Exception as erro:
            return False, f"Falha ao criar backup do dataset: {erro}"

        # 2. Carrega dataset JSON atual
        try:
            with open(self.caminho_intencoes, mode="r", encoding="utf-8") as f:
                dataset = json.load(f)
        except Exception as erro:
            return False, f"Erro ao ler {self.caminho_intencoes.name}: {erro}"

        # 3. Localiza a intenção e insere a nova frase
        intencao_encontrada = False
        for intencao in dataset.get("intents", []):
            if intencao["tag"] == tag_intencao:
                intencao_encontrada = True
                padroes = intencao.setdefault("patterns", [])
                # Evita duplicatas case-insensitive
                if not any(p.lower() == frase_limpa.lower() for p in padroes):
                    padroes.append(frase_limpa)
                break

        if not intencao_encontrada:
            return False, f"Intenção '{tag_intencao}' não encontrada no dataset."

        # 4. Grava dataset atualizado
        try:
            with open(self.caminho_intencoes, mode="w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
                f.flush()
        except Exception as erro:
            return False, f"Erro ao salvar {self.caminho_intencoes.name} atualizado: {erro}"

        # 5. Marca mensagem no log como revisada (JSONL + CSV)
        if id_interacao:
            self.marcar_log_revisado(id_interacao)

        # 6. Retreinamento em tempo de execução (Hot Retrain)
        try:
            self.motor.carregar_e_treinar()
        except Exception as erro:
            return False, f"Frase salva, mas erro ao retreinar modelo: {erro}"

        return True, f"Frase adicionada com sucesso à intenção '{tag_intencao}'. Modelo retreinado!"

    def criar_nova_intencao(
        self,
        tag: str,
        categoria: str,
        descricao: str,
        primeiro_padrao: str,
        primeira_resposta: str,
        id_interacao: str | None = None
    ) -> tuple[bool, str]:
        """Cria uma nova categoria de intenção no dataset e retreina o classificador."""
        tag_limpa = tag.strip().lower().replace(" ", "_")
        if not tag_limpa:
            return False, "A tag da intenção não pode ser vazia."

        try:
            self.criar_backup()
        except Exception as erro:
            return False, f"Falha ao criar backup do dataset: {erro}"

        try:
            with open(self.caminho_intencoes, mode="r", encoding="utf-8") as f:
                dataset = json.load(f)
        except Exception as erro:
            return False, f"Erro ao ler {self.caminho_intencoes.name}: {erro}"

        # Verifica se já existe
        for intencao in dataset.get("intents", []):
            if intencao["tag"] == tag_limpa:
                return False, f"A intenção '{tag_limpa}' já existe no dataset."

        novo_bloco_intencao = {
            "tag": tag_limpa,
            "categoria": categoria or "tecnica",
            "descricao": descricao or f"Intenção criada dinamicamente: {tag_limpa}",
            "patterns": [primeiro_padrao.strip()],
            "responses": [primeira_resposta.strip()],
        }

        dataset.setdefault("intents", []).append(novo_bloco_intencao)

        try:
            with open(self.caminho_intencoes, mode="w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
                f.flush()
        except Exception as erro:
            return False, f"Erro ao salvar {self.caminho_intencoes.name}: {erro}"

        if id_interacao:
            self.marcar_log_revisado(id_interacao)

        try:
            self.motor.carregar_e_treinar()
        except Exception as erro:
            return False, f"Intenção criada, mas erro ao retreinar modelo: {erro}"

        return True, f"Nova intenção '{tag_limpa}' criada com sucesso. Modelo retreinado!"
