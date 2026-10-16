"""Módulo Didático de Avaliação Quantitativa e Métricas de Machine Learning.

Referência Didática:
    - Aula 06 do Plano de Ensino (CETAM / IBC).
    - Métricas de classificação supervisionada: Acurácia, Precisão, Recall e F1-Score.
    - Análise de Matriz de Confusão e por que a acurácia pode enganar em classes desbalanceadas.
    - Consolidação de métricas do modelo e métricas operacionais para o comando /metricas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.ml_engine import MLEngine
from src.privacy_logger import PrivacyLogger


class ModelEvaluator:
    """Avaliador educacional de métricas de ML e análise de tráfego de conversas."""

    def __init__(
        self,
        ml_engine: MLEngine | None = None,
        privacy_logger: PrivacyLogger | None = None
    ) -> None:
        self.engine: MLEngine = ml_engine or MLEngine()
        self.logger: PrivacyLogger = privacy_logger or PrivacyLogger()

    def evaluate_models(self) -> Dict[str, Any]:
        """Calcula as métricas de validação cruzada para Regressão Logística e Naive Bayes."""
        if not self.engine.is_trained:
            self.engine.load_and_train()

        X_sentences = self.engine.training_sentences
        y_labels = np.array(self.engine.training_labels)

        # Vetorização de todo o dataset
        X_vec = self.engine.vectorizer.fit_transform(X_sentences)

        # Ajusta número de folds com base no menor número de amostras por classe
        unique_classes, counts = np.unique(y_labels, return_counts=True)
        min_samples = int(np.min(counts))
        n_splits = max(2, min(3, min_samples))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 1. Validação Cruzada - Regressão Logística
        y_pred_lr = cross_val_predict(self.engine.clf_logistic, X_vec, y_labels, cv=cv)
        acc_lr = float(accuracy_score(y_labels, y_pred_lr))
        prec_lr = float(precision_score(y_labels, y_pred_lr, average="weighted", zero_division=0))
        rec_lr = float(recall_score(y_labels, y_pred_lr, average="weighted", zero_division=0))
        f1_lr = float(f1_score(y_labels, y_pred_lr, average="weighted", zero_division=0))

        # 2. Validação Cruzada - Naive Bayes
        y_pred_nb = cross_val_predict(self.engine.clf_nb, X_vec, y_labels, cv=cv)
        acc_nb = float(accuracy_score(y_labels, y_pred_nb))
        prec_nb = float(precision_score(y_labels, y_pred_nb, average="weighted", zero_division=0))
        rec_nb = float(recall_score(y_labels, y_pred_nb, average="weighted", zero_division=0))
        f1_nb = float(f1_score(y_labels, y_pred_nb, average="weighted", zero_division=0))

        # 3. Análise dos maiores erros na Matriz de Confusão (Regressão Logística)
        classes_list = list(self.engine.clf_logistic.classes_)
        cm = confusion_matrix(y_labels, y_pred_lr, labels=classes_list)

        confusions: List[Tuple[str, str, int]] = []
        for i, true_cls in enumerate(classes_list):
            for j, pred_cls in enumerate(classes_list):
                if i != j and cm[i, j] > 0:
                    confusions.append((true_cls, pred_cls, int(cm[i, j])))

        confusions.sort(key=lambda item: item[2], reverse=True)

        return {
            "total_amostras": len(X_sentences),
            "total_intencoes": len(unique_classes),
            "n_folds": n_splits,
            "logistic_regression": {
                "acuracia": round(acc_lr * 100, 2),
                "precisao": round(prec_lr * 100, 2),
                "recall": round(rec_lr * 100, 2),
                "f1_score": round(f1_lr * 100, 2),
            },
            "naive_bayes": {
                "acuracia": round(acc_nb * 100, 2),
                "precisao": round(prec_nb * 100, 2),
                "recall": round(rec_nb * 100, 2),
                "f1_score": round(f1_nb * 100, 2),
            },
            "principais_confusoes": confusions[:5]
        }

    def format_report_text(self) -> str:
        """Gera um relatório completo e didático adequado para envio no Telegram ou Terminal."""
        model_metrics = self.evaluate_models()
        ops_metrics = self.logger.get_metrics_summary()

        lr = model_metrics["logistic_regression"]
        nb = model_metrics["naive_bayes"]

        report_lines = [
            "📊 **RELATÓRIO DE MÉTRICAS PEDAGÓGICAS (PLN / ML)**",
            f"• Amostras no Dataset: {model_metrics['total_amostras']} frases",
            f"• Classes (Intenções): {model_metrics['total_intencoes']} categorias",
            f"• Validação Cruzada: {model_metrics['n_folds']}-Fold Stratified\n",
            "🔹 **1. Desempenho do Modelo Principal (Regressão Logística):**",
            f"  - Acurácia Geral: {lr['acuracia']}%",
            f"  - Precisão Ponderada: {lr['precisao']}%",
            f"  - Recall (Revocação): {lr['recall']}%",
            f"  - F1-Score: {lr['f1_score']}%\n",
            "🔹 **2. Comparativo Didático (Naive Bayes Multinomial):**",
            f"  - Acurácia: {nb['acuracia']}% | F1-Score: {nb['f1_score']}%\n",
            "🔹 **3. Métricas Operacionais de Uso (Logs Anonimizados):**",
            f"  - Total de Mensagens: {ops_metrics['total_interacoes']}",
            f"  - Taxa de Resolução: {ops_metrics['taxa_resolucao']}%",
            f"  - Taxa de Fallback: {ops_metrics['taxa_fallback']}%",
            f"  - Confiança Média: {ops_metrics['confianca_media']}",
            f"  - Alunos Únicos: {ops_metrics['usuarios_unicos']}"
        ]

        # Principais confusões identificadas
        if model_metrics["principais_confusoes"]:
            report_lines.append("\n⚠️ **Principais Dúvidas da Matriz de Confusão:**")
            for true_c, pred_c, count in model_metrics["principais_confusoes"]:
                report_lines.append(f"  • Era '{true_c}', mas previu '{pred_c}' ({count}x)")
        else:
            report_lines.append("\n✅ Nenhuma confusão grave detectada no dataset atual!")

        # Top assuntos perguntados
        if ops_metrics["top_intencoes"]:
            report_lines.append("\n📌 **Assuntos mais perguntados:**")
            for intent, count in ops_metrics["top_intencoes"]:
                report_lines.append(f"  • {intent}: {count} perguntas")

        return "\n".join(report_lines)
