"""Testes unitários para o motor de Machine Learning e Fallback."""

from src.motor_ml import MLEngine


def test_model_training_and_intent_recognition():
    engine = MLEngine()
    assert engine.is_trained
    assert len(engine.training_sentences) > 0

    res = engine.predict("Quem é o professor da matéria?")
    assert res.intent == "professor_info"
    assert not res.is_fallback
    assert "Agnaldo Marcelo Pereira Fournier" in res.response


def test_technical_question_recognition():
    engine = MLEngine()
    res = engine.predict("O que é TF-IDF?")
    assert res.intent == "representacao_tfidf_embeddings"
    assert not res.is_fallback
    assert res.confidence >= 0.35


def test_fallback_on_unrelated_text():
    engine = MLEngine()
    res = engine.predict("qual a temperatura do sol hoje de tarde?")
    assert res.is_fallback
    assert res.intent == "fallback"
