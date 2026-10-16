"""Motor de Aprendizado de Máquina Clássico Supervisionado e Fallback.

Referência Didática:
    - Aulas 04, 05 e 07 do Plano de Ensino (CETAM / IBC).
    - Vetorização numérica com TF-IDF (Term Frequency - Inverse Document Frequency)
      unigramas e bigramas.
    - Classificação de intenções com Regressão Logística (principal) e
      Multinomial Naive Bayes (comparativo didático).
    - Mecanismo de corte por limiar probabilístico (Fallback).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.config import CONFIDENCE_THRESHOLD, DEFAULT_ML_MODEL, INTENTS_FILE
from src.nlp_pipeline import NLPPipeline

# Respostas padrão para quando a confiança do modelo for inferior ao limiar
FALLBACK_RESPONSES = [
    (
        "Desculpe, não consegui entender com clareza a sua dúvida sobre PLN. "
        "Você poderia reformular a pergunta ou tentar com outras palavras? "
        "Se quiser ver os tópicos que domino, digite /ajuda."
    ),
    (
        "Ainda estou aprendendo e minha certeza para essa pergunta foi baixa. "
        "Tente perguntar sobre critérios de notas, o plano de aula, ou conceitos "
        "como Tokenização, TF-IDF, Naive Bayes e Regressão Logística. "
        "Você também pode usar o comando /ajuda."
    ),
    (
        "Hum, não tenho certeza se compreendi o que você precisa. "
        "Tente simplificar a frase ou verifique a ortografia. "
        "Essa sua pergunta foi registrada e poderá me ajudar a aprender no futuro!"
    )
]


@dataclass
class PredictionResult:
    """Estrutura com os resultados completos da inferência probabilística."""
    intent: str
    confidence: float
    is_fallback: bool
    response: str
    top_candidates: List[Tuple[str, float]] = field(default_factory=list)
    nb_intent: str = ""
    nb_confidence: float = 0.0
    active_tfidf_terms: List[Tuple[str, float]] = field(default_factory=list)
    processed_text: str = ""
    primary_model: str = "LogisticRegression"


class MLEngine:
    """Motor didático de vetorização e classificação probabilística."""

    def __init__(
        self,
        intents_path: Path | None = None,
        nlp_pipeline: NLPPipeline | None = None
    ) -> None:
        self.intents_path: Path = intents_path or INTENTS_FILE
        self.nlp: NLPPipeline = nlp_pipeline or NLPPipeline()
        self.primary_model_name: str = DEFAULT_ML_MODEL

        # Componentes Scikit-Learn
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Considera palavras isoladas e pares contíguos (Bigramas)
            sublinear_tf=True,   # Aplica escala logarítmica à contagem para suavizar termos repetidos
            strip_accents=None   # A normalização já é feita de forma controlada no nlp_pipeline
        )
        self.clf_logistic: LogisticRegression = LogisticRegression(
            C=10.0,              # Regularização ajustada para vocabulários textuais esparsos
            max_iter=500,
            random_state=42
        )
        self.clf_nb: MultinomialNB = MultinomialNB(alpha=0.1)

        # Dados em memória
        self.raw_data: Dict[str, Any] = {}
        self.responses_by_intent: Dict[str, List[str]] = {}
        self.descriptions_by_intent: Dict[str, str] = {}
        self.training_sentences: List[str] = []
        self.training_labels: List[str] = []
        self.is_trained: bool = False

        # Inicializa e treina na carga
        self.load_and_train()

    def load_and_train(self) -> None:
        """Lê o dataset JSON externo e executa o treinamento imediato em memória."""
        if not self.intents_path.exists():
            raise FileNotFoundError(
                f"Arquivo de intenções não encontrado em: {self.intents_path}"
            )

        with open(self.intents_path, mode="r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        sentences: List[str] = []
        labels: List[str] = []
        self.responses_by_intent.clear()
        self.descriptions_by_intent.clear()

        for intent in self.raw_data.get("intents", []):
            tag = intent["tag"]
            self.responses_by_intent[tag] = intent.get("responses", [])
            self.descriptions_by_intent[tag] = intent.get("descricao", "")

            for pattern in intent.get("patterns", []):
                processed = self.nlp.process(pattern)
                if processed.strip():
                    sentences.append(processed)
                    labels.append(tag)

        if not sentences:
            raise ValueError("O dataset de intenções não contém padrões de frases válidos para treino.")

        self.training_sentences = sentences
        self.training_labels = labels

        # Ajuste do Vetorizador TF-IDF
        X_tfidf = self.vectorizer.fit_transform(self.training_sentences)
        y = np.array(self.training_labels)

        # Treinamento dos dois modelos
        self.clf_logistic.fit(X_tfidf, y)
        self.clf_nb.fit(X_tfidf, y)

        self.is_trained = True

    def predict(
        self,
        text: str,
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> PredictionResult:
        """Classifica uma frase do usuário, calculando probabilidades e ativando fallback."""
        if not self.is_trained:
            self.load_and_train()

        # 1. Pré-processamento
        processed_text = self.nlp.process(text)
        if not processed_text.strip():
            # Frase vazia ou apenas ruído/pontuação
            return PredictionResult(
                intent="fallback",
                confidence=0.0,
                is_fallback=True,
                response=random.choice(FALLBACK_RESPONSES),
                processed_text=processed_text
            )

        # 2. Vetorização numérica da frase
        X_vec = self.vectorizer.transform([processed_text])

        # Extração dos termos TF-IDF com seus pesos para inspeção no /debug
        feature_names = self.vectorizer.get_feature_names_out()
        cx = X_vec.tocoo()
        active_terms = [
            (feature_names[col], float(val))
            for col, val in zip(cx.col, cx.data)
        ]
        active_terms.sort(key=lambda item: item[1], reverse=True)

        # 3. Predição probabilística - Regressão Logística
        proba_lr = self.clf_logistic.predict_proba(X_vec)[0]
        classes_lr = self.clf_logistic.classes_
        sorted_lr = np.argsort(proba_lr)[::-1]
        top_candidates_lr = [
            (str(classes_lr[idx]), float(proba_lr[idx]))
            for idx in sorted_lr[:3]
        ]

        # 4. Predição comparativa - Naive Bayes
        proba_nb = self.clf_nb.predict_proba(X_vec)[0]
        classes_nb = self.clf_nb.classes_
        sorted_nb = np.argsort(proba_nb)[::-1]
        top_candidates_nb = [
            (str(classes_nb[idx]), float(proba_nb[idx]))
            for idx in sorted_nb[:3]
        ]
        nb_best_intent, nb_best_prob = top_candidates_nb[0]

        # Seleciona qual modelo comanda a decisão primária
        if self.primary_model_name.lower() in ("multinomialnb", "nb", "naivebayes"):
            primary_intent, primary_prob = top_candidates_nb[0]
            top_candidates = top_candidates_nb
            active_model = "MultinomialNB"
        else:
            primary_intent, primary_prob = top_candidates_lr[0]
            top_candidates = top_candidates_lr
            active_model = "LogisticRegression"

        # 5. Avaliação do limiar de Fallback
        if primary_prob < threshold:
            return PredictionResult(
                intent="fallback",
                confidence=primary_prob,
                is_fallback=True,
                response=random.choice(FALLBACK_RESPONSES),
                top_candidates=top_candidates,
                nb_intent=nb_best_intent,
                nb_confidence=nb_best_prob,
                active_tfidf_terms=active_terms,
                processed_text=processed_text,
                primary_model=active_model
            )

        # 6. Seleção de resposta da intenção reconhecida
        possible_responses = self.responses_by_intent.get(primary_intent, [])
        selected_response = (
            random.choice(possible_responses)
            if possible_responses
            else "Entendi a sua pergunta, mas ainda não tenho uma resposta cadastrada para esse assunto."
        )

        return PredictionResult(
            intent=primary_intent,
            confidence=primary_prob,
            is_fallback=False,
            response=selected_response,
            top_candidates=top_candidates,
            nb_intent=nb_best_intent,
            nb_confidence=nb_best_prob,
            active_tfidf_terms=active_terms,
            processed_text=processed_text,
            primary_model=active_model
        )

    def get_all_intents(self) -> List[str]:
        """Retorna a lista de todas as tags de intenção conhecidas."""
        return list(self.responses_by_intent.keys())
