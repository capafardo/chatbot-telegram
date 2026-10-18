"""Motor de Aprendizado de Máquina Clássico Supervisionado e Fallback.

Referência Didática:
    - Aulas 04, 05 e 07 do Plano de Ensino (CETAM / IBC).
    - Vetorização numérica com TF-IDF (Term Frequency - Inverse Document Frequency)
      unigramas e bigramas.
    - Classificação de intenções com Regressão Logística (principal) e
      Multinomial Naive Bayes (comparativo didático).
    - Mecanismo de corte por limiar probabilístico (Fallback).
    - Suporte a personalização "White-Label" via metadados do dataset JSON.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.configuracao import ARQUIVO_INTENCOES, LIMIAR_CONFIANCA, MODELO_ML_PADRAO
from src.pipeline_pln import PipelinePLN

# Respostas de Fallback genéricas (agnósticas de assunto), usadas apenas se o arquivo
# intencoes.json não definir suas próprias 'respostas_fallback' em 'metadata'.
RESPOSTAS_FALLBACK_PADRAO: List[str] = [
    (
        "Desculpe, não consegui entender com clareza a sua pergunta. "
        "Você poderia reformular ou tentar com outras palavras? "
        "Digite /ajuda para ver os comandos e exemplos de tópicos que conheço."
    ),
    (
        "Ainda estou aprendendo e minha certeza para essa pergunta foi baixa. "
        "Tente escrever de forma mais direta ou digite /ajuda para ver sugestões de perguntas."
    ),
    (
        "Hum, não tenho certeza se compreendi o que você precisa. "
        "Tente simplificar a frase. Essa sua dúvida foi registrada e poderá me ajudar a aprender no futuro!"
    )
]


@dataclass
class ResultadoPredicao:
    """Estrutura com os resultados completos da inferência probabilística."""
    intencao: str
    confianca: float
    acionou_fallback: bool
    resposta: str
    principais_candidatos: List[Tuple[str, float]] = field(default_factory=list)
    intencao_nb: str = ""
    confianca_nb: float = 0.0
    termos_tfidf_ativos: List[Tuple[str, float]] = field(default_factory=list)
    texto_processado: str = ""
    modelo_primario: str = "LogisticRegression"


class MotorML:
    """Motor didático de vetorização e classificação probabilística."""

    def __init__(
        self,
        caminho_intencoes: Path | None = None,
        pipeline_pln: PipelinePLN | None = None
    ) -> None:
        self.caminho_intencoes: Path = caminho_intencoes or ARQUIVO_INTENCOES
        self.pln: PipelinePLN = pipeline_pln or PipelinePLN()
        self.nome_modelo_primario: str = MODELO_ML_PADRAO

        # Componentes Scikit-Learn
        self.vetorizador: TfidfVectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",  # Preserva dígitos isolados (ex: 'aula 1') e termos de 1 caractere
            ngram_range=(1, 2),            # Considera palavras isoladas e pares contíguos (Bigramas)
            sublinear_tf=True,             # Escala logarítmica para suavizar termos repetidos
            strip_accents=None             # A normalização de acentos já é controlada no PipelinePLN
        )
        self.classificador_logistico: LogisticRegression = LogisticRegression(
            C=10.0,                        # Regularização ajustada para vocabulários textuais esparsos
            max_iter=500,
            random_state=42
        )
        self.classificador_nb: MultinomialNB = MultinomialNB(alpha=0.1)

        # Dados em memória
        self.dados_brutos: Dict[str, Any] = {}
        self.metadados: Dict[str, Any] = {}
        self.respostas_por_intencao: Dict[str, List[str]] = {}
        self.descricoes_por_intencao: Dict[str, str] = {}
        self.respostas_fallback: List[str] = list(RESPOSTAS_FALLBACK_PADRAO)
        self.frases_treinamento: List[str] = []
        self.rotulos_treinamento: List[str] = []
        self.esta_treinado: bool = False
        self.versao_treino: int = 0

        # Inicializa e treina na carga
        self.carregar_e_treinar()

    def carregar_e_treinar(self) -> None:
        """Lê o dataset JSON externo, extrai metadados e treina os modelos em memória."""
        if not self.caminho_intencoes.exists():
            raise FileNotFoundError(
                f"Arquivo de intenções não encontrado em: {self.caminho_intencoes}"
            )

        with open(self.caminho_intencoes, mode="r", encoding="utf-8") as f:
            self.dados_brutos = json.load(f)

        self.metadados = self.dados_brutos.get("metadata", {})
        fallbacks_customizados = self.metadados.get("respostas_fallback", [])
        if isinstance(fallbacks_customizados, list) and fallbacks_customizados:
            self.respostas_fallback = [str(r) for r in fallbacks_customizados if str(r).strip()]
        else:
            self.respostas_fallback = list(RESPOSTAS_FALLBACK_PADRAO)

        frases: List[str] = []
        rotulos: List[str] = []
        self.respostas_por_intencao.clear()
        self.descricoes_por_intencao.clear()

        for intencao in self.dados_brutos.get("intents", []):
            tag = intencao["tag"]
            self.respostas_por_intencao[tag] = intencao.get("responses", [])
            self.descricoes_por_intencao[tag] = intencao.get("descricao", "")

            for padrao in intencao.get("patterns", []):
                processado = self.pln.processar(padrao)
                if processado.strip():
                    frases.append(processado)
                    rotulos.append(tag)

        if not frases:
            raise ValueError("O dataset de intenções não contém padrões de frases válidos para treino.")

        self.frases_treinamento = frases
        self.rotulos_treinamento = rotulos

        # Ajuste do Vetorizador TF-IDF
        matriz_tfidf = self.vetorizador.fit_transform(self.frases_treinamento)
        vetor_y = np.array(self.rotulos_treinamento)

        # Treinamento dos dois modelos
        self.classificador_logistico.fit(matriz_tfidf, vetor_y)
        self.classificador_nb.fit(matriz_tfidf, vetor_y)

        self.esta_treinado = True
        self.versao_treino += 1

    def prever(
        self,
        texto: str,
        limiar: float = LIMIAR_CONFIANCA
    ) -> ResultadoPredicao:
        """Classifica uma frase do usuário, calculando probabilidades e ativando fallback."""
        if not self.esta_treinado:
            self.carregar_e_treinar()

        # 1. Pré-processamento
        texto_processado = self.pln.processar(texto)
        if not texto_processado.strip():
            return ResultadoPredicao(
                intencao="fallback",
                confianca=0.0,
                acionou_fallback=True,
                resposta=random.choice(self.respostas_fallback),
                texto_processado=texto_processado
            )

        # 2. Vetorização numérica da frase
        vetor_x = self.vetorizador.transform([texto_processado])

        # Extração dos termos TF-IDF com seus pesos para inspeção no /debug
        nomes_features = self.vetorizador.get_feature_names_out()
        coordenadas = vetor_x.tocoo()
        termos_ativos = [
            (str(nomes_features[col]), float(val))
            for col, val in zip(coordenadas.col, coordenadas.data)
        ]
        termos_ativos.sort(key=lambda item: item[1], reverse=True)

        # 3. Predição probabilística - Regressão Logística
        probabilidades_lr = self.classificador_logistico.predict_proba(vetor_x)[0]
        classes_lr = self.classificador_logistico.classes_
        indices_lr = np.argsort(probabilidades_lr)[::-1]
        candidatos_lr = [
            (str(classes_lr[idx]), float(probabilidades_lr[idx]))
            for idx in indices_lr[:3]
        ]

        # 4. Predição comparativa - Naive Bayes
        probabilidades_nb = self.classificador_nb.predict_proba(vetor_x)[0]
        classes_nb = self.classificador_nb.classes_
        indices_nb = np.argsort(probabilidades_nb)[::-1]
        candidatos_nb = [
            (str(classes_nb[idx]), float(probabilidades_nb[idx]))
            for idx in indices_nb[:3]
        ]
        melhor_intencao_nb, melhor_prob_nb = candidatos_nb[0]

        # Seleciona qual modelo comanda a decisão primária
        if self.nome_modelo_primario.lower() in ("multinomialnb", "nb", "naivebayes"):
            intencao_primaria, prob_primaria = candidatos_nb[0]
            principais_candidatos = candidatos_nb
            modelo_ativo = "MultinomialNB"
        else:
            intencao_primaria, prob_primaria = candidatos_lr[0]
            principais_candidatos = candidatos_lr
            modelo_ativo = "LogisticRegression"

        # 5. Avaliação do limiar de Fallback
        if prob_primaria < limiar:
            return ResultadoPredicao(
                intencao="fallback",
                confianca=prob_primaria,
                acionou_fallback=True,
                resposta=random.choice(self.respostas_fallback),
                principais_candidatos=principais_candidatos,
                intencao_nb=melhor_intencao_nb,
                confianca_nb=melhor_prob_nb,
                termos_tfidf_ativos=termos_ativos,
                texto_processado=texto_processado,
                modelo_primario=modelo_ativo
            )

        # 6. Seleção de resposta da intenção reconhecida
        respostas_possiveis = self.respostas_por_intencao.get(intencao_primaria, [])
        resposta_selecionada = (
            random.choice(respostas_possiveis)
            if respostas_possiveis
            else "Entendi a sua pergunta, mas ainda não tenho uma resposta cadastrada para esse assunto."
        )

        return ResultadoPredicao(
            intencao=intencao_primaria,
            confianca=prob_primaria,
            acionou_fallback=False,
            resposta=resposta_selecionada,
            principais_candidatos=principais_candidatos,
            intencao_nb=melhor_intencao_nb,
            confianca_nb=melhor_prob_nb,
            termos_tfidf_ativos=termos_ativos,
            texto_processado=texto_processado,
            modelo_primario=modelo_ativo
        )

    def obter_todas_intencoes(self) -> List[str]:
        """Retorna a lista de todas as tags de intenção conhecidas."""
        return list(self.respostas_por_intencao.keys())

    def obter_metadados_bot(self) -> Dict[str, Any]:
        """Retorna os metadados de identidade e configuração do bot definidos no dataset JSON."""
        return dict(self.metadados)
