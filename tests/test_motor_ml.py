"""Testes unitários para o MotorML e mecanismo de Fallback."""

from src.motor_ml import MotorML


def test_treinamento_modelo_e_reconhecimento_intencao():
    motor = MotorML()
    assert motor.esta_treinado
    assert len(motor.frases_treinamento) > 0

    resultado = motor.prever("Quem é o professor da matéria?")
    assert resultado.intencao == "professor_info"
    assert not resultado.acionou_fallback
    assert "Agnaldo Marcelo Pereira Fournier" in resultado.resposta


def test_reconhecimento_pergunta_tecnica_com_e_sem_acento():
    motor = MotorML()
    res1 = motor.prever("O que é TF-IDF?")
    assert res1.intencao == "representacao_tfidf_embeddings"
    assert not res1.acionou_fallback
    assert res1.confianca >= 0.35

    res2 = motor.prever("como funciona a regressao logistica sem acento?")
    assert not res2.acionou_fallback


def test_fallback_em_texto_fora_de_escopo():
    motor = MotorML()
    resultado = motor.prever("qual a temperatura do sol hoje de tarde?")
    assert resultado.acionou_fallback
    assert resultado.intencao == "fallback"
