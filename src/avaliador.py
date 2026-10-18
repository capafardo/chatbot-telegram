"""Módulo Didático de Avaliação Quantitativa e Métricas de Machine Learning.

Referência Didática:
    - Aula 06 do Plano de Ensino (CETAM / IBC).
    - Métricas de classificação supervisionada: Acurácia, Precisão, Recall e F1-Score.
    - Análise de Matriz de Confusão e por que a acurácia pode enganar em classes desbalanceadas.
    - Uso de Pipeline no StratifiedKFold para evitar vazamento de dados (Data Leakage).
    - Proteção contra classes recém-criadas com apenas 1 amostra e cache em memória.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.motor_ml import MotorML
from src.registrador_privacidade import RegistradorPrivacidade


class AvaliadorModelo:
    """Avaliador educacional de métricas de ML e análise de tráfego de conversas."""

    def __init__(
        self,
        motor_ml: MotorML | None = None,
        registrador_privacidade: RegistradorPrivacidade | None = None
    ) -> None:
        self.motor: MotorML = motor_ml or MotorML()
        self.registrador: RegistradorPrivacidade = registrador_privacidade or RegistradorPrivacidade()

        # Cache em memória das métricas de validação cruzada (evita sobrecarga de CPU)
        self._versao_cache: int = -1
        self._metricas_modelo_cache: Optional[Dict[str, Any]] = None

    def avaliar_modelos(self) -> Dict[str, Any]:
        """Calcula as métricas de validação cruzada para Regressão Logística e Naive Bayes."""
        if not self.motor.esta_treinado:
            self.motor.carregar_e_treinar()

        # Retorna do cache se o modelo não sofreu retreinamento desde a última avaliação
        if (
            self._metricas_modelo_cache is not None
            and self._versao_cache == self.motor.versao_treino
        ):
            return self._metricas_modelo_cache

        frases_todas = self.motor.frases_treinamento
        rotulos_todos = np.array(self.motor.rotulos_treinamento)

        # Identifica classes que possuem pelo menos 2 amostras (requisito matemático do K-Fold >= 2)
        classes_unicas, contagens = np.unique(rotulos_todos, return_counts=True)
        classes_validas = set(classes_unicas[contagens >= 2])
        classes_insuficientes = [str(c) for c in classes_unicas[contagens < 2]]

        if len(classes_validas) < 2:
            # Caso extremo: dataset minúsculo de teste com menos de 2 classes válidas
            resultado_minimo: Dict[str, Any] = {
                "total_amostras": len(frases_todas),
                "total_intencoes": len(classes_unicas),
                "intencoes_poucas_amostras": classes_insuficientes,
                "n_folds": 0,
                "regressao_logistica": {
                    "acuracia": 100.0,
                    "precisao": 100.0,
                    "recall": 100.0,
                    "f1_score": 100.0,
                },
                "naive_bayes": {
                    "acuracia": 100.0,
                    "precisao": 100.0,
                    "recall": 100.0,
                    "f1_score": 100.0,
                },
                "principais_confusoes": [],
            }
            self._versao_cache = self.motor.versao_treino
            self._metricas_modelo_cache = resultado_minimo
            return resultado_minimo

        # Filtra apenas amostras de classes com pelo menos 2 exemplos para o StratifiedKFold
        mascara_valida = np.isin(rotulos_todos, list(classes_validas))
        frases_cv = [frase for frase, ok in zip(frases_todas, mascara_valida) if ok]
        rotulos_cv = rotulos_todos[mascara_valida]

        _, contagens_cv = np.unique(rotulos_cv, return_counts=True)
        min_amostras = int(np.min(contagens_cv))
        n_divisoes = max(2, min(3, min_amostras))

        validacao_cruzada = StratifiedKFold(
            n_splits=n_divisoes, shuffle=True, random_state=42
        )

        # Pipeline isolado (clone) para evitar vazamento de dados (Data Leakage) e não mutar self.motor.vetorizador
        pipeline_lr = Pipeline([
            ("tfidf", clone(self.motor.vetorizador)),
            ("clf", clone(self.motor.classificador_logistico)),
        ])
        pipeline_nb = Pipeline([
            ("tfidf", clone(self.motor.vetorizador)),
            ("clf", clone(self.motor.classificador_nb)),
        ])

        # 1. Validação Cruzada - Regressão Logística
        pred_lr = cross_val_predict(pipeline_lr, frases_cv, rotulos_cv, cv=validacao_cruzada)
        acuracia_lr = float(accuracy_score(rotulos_cv, pred_lr))
        precisao_lr = float(precision_score(rotulos_cv, pred_lr, average="weighted", zero_division=0))
        recall_lr = float(recall_score(rotulos_cv, pred_lr, average="weighted", zero_division=0))
        f1_lr = float(f1_score(rotulos_cv, pred_lr, average="weighted", zero_division=0))

        # 2. Validação Cruzada - Naive Bayes
        pred_nb = cross_val_predict(pipeline_nb, frases_cv, rotulos_cv, cv=validacao_cruzada)
        acuracia_nb = float(accuracy_score(rotulos_cv, pred_nb))
        precisao_nb = float(precision_score(rotulos_cv, pred_nb, average="weighted", zero_division=0))
        recall_nb = float(recall_score(rotulos_cv, pred_nb, average="weighted", zero_division=0))
        f1_nb = float(f1_score(rotulos_cv, pred_nb, average="weighted", zero_division=0))

        # 3. Análise dos maiores erros na Matriz de Confusão (Regressão Logística)
        lista_classes = sorted(list(classes_validas))
        matriz_confusao = confusion_matrix(rotulos_cv, pred_lr, labels=lista_classes)

        confusoes: List[Tuple[str, str, int]] = []
        for i, classe_real in enumerate(lista_classes):
            for j, classe_predita in enumerate(lista_classes):
                if i != j and matriz_confusao[i, j] > 0:
                    confusoes.append((str(classe_real), str(classe_predita), int(matriz_confusao[i, j])))

        confusoes.sort(key=lambda item: item[2], reverse=True)

        resultado: Dict[str, Any] = {
            "total_amostras": len(frases_todas),
            "total_intencoes": len(classes_unicas),
            "intencoes_poucas_amostras": classes_insuficientes,
            "n_folds": n_divisoes,
            "regressao_logistica": {
                "acuracia": round(acuracia_lr * 100, 2),
                "precisao": round(precisao_lr * 100, 2),
                "recall": round(recall_lr * 100, 2),
                "f1_score": round(f1_lr * 100, 2),
            },
            "naive_bayes": {
                "acuracia": round(acuracia_nb * 100, 2),
                "precisao": round(precisao_nb * 100, 2),
                "recall": round(recall_nb * 100, 2),
                "f1_score": round(f1_nb * 100, 2),
            },
            "principais_confusoes": confusoes[:5],
        }

        self._versao_cache = self.motor.versao_treino
        self._metricas_modelo_cache = resultado
        return resultado

    def formatar_relatorio_texto(self) -> str:
        """Gera um relatório completo e didático adequado para envio no Telegram ou Terminal."""
        metricas_modelo = self.avaliar_modelos()
        metricas_operacionais = self.registrador.obter_resumo_metricas()

        lr = metricas_modelo["regressao_logistica"]
        nb = metricas_modelo["naive_bayes"]

        linhas_relatorio = [
            "📊 **RELATÓRIO DE MÉTRICAS PEDAGÓGICAS (PLN / ML)**",
            f"• Amostras no Dataset: {metricas_modelo['total_amostras']} frases",
            f"• Classes (Intenções): {metricas_modelo['total_intencoes']} categorias",
            f"• Validação Cruzada: {metricas_modelo['n_folds']}-Fold Stratified\n",
            "🔹 **1. Desempenho do Modelo Principal (Regressão Logística):**",
            f"  - Acurácia Geral: {lr['acuracia']}%",
            f"  - Precisão Ponderada: {lr['precisao']}%",
            f"  - Recall (Revocação): {lr['recall']}%",
            f"  - F1-Score: {lr['f1_score']}%\n",
            "🔹 **2. Comparativo Didático (Naive Bayes Multinomial):**",
            f"  - Acurácia: {nb['acuracia']}% | F1-Score: {nb['f1_score']}%\n",
            "🔹 **3. Métricas Operacionais de Uso (Logs Anonimizados):**",
            f"  - Total de Mensagens: {metricas_operacionais['total_interacoes']}",
            f"  - Taxa de Resolução: {metricas_operacionais['taxa_resolucao']}%",
            f"  - Taxa de Fallback: {metricas_operacionais['taxa_fallback']}%",
            f"  - Confiança Média: {metricas_operacionais['confianca_media']}",
            f"  - Usuários Únicos: {metricas_operacionais['usuarios_unicos']}",
        ]

        if metricas_modelo.get("intencoes_poucas_amostras"):
            poucas = ", ".join(metricas_modelo["intencoes_poucas_amostras"])
            linhas_relatorio.append(
                f"\nℹ️ *Nota: Intenções recém-criadas com apenas 1 frase ({poucas}) "
                f"aguardam novas amostras para entrar na divisão K-Fold.*"
            )

        # Principais confusões identificadas
        if metricas_modelo["principais_confusoes"]:
            linhas_relatorio.append("\n⚠️ **Principais Dúvidas da Matriz de Confusão:**")
            for real_c, pred_c, cont in metricas_modelo["principais_confusoes"]:
                linhas_relatorio.append(f"  • Era `{real_c}`, mas previu `{pred_c}` ({cont}x)")
        else:
            linhas_relatorio.append("\n✅ Nenhuma confusão grave detectada no dataset atual!")

        # Top assuntos perguntados
        if metricas_operacionais["top_intencoes"]:
            linhas_relatorio.append("\n📌 **Assuntos mais perguntados:**")
            for intencao, cont in metricas_operacionais["top_intencoes"]:
                linhas_relatorio.append(f"  • `{intencao}`: {cont} pergunta(s)")

        return "\n".join(linhas_relatorio)
