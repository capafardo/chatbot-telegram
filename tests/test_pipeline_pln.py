"""Testes unitários para o pipeline de PLN."""

from src.pipeline_pln import NLPPipeline


def test_slang_expansion():
    pipe = NLPPipeline()
    text = "vc vai na aula hj prof?"
    expanded = pipe.expand_slang(text.lower())
    assert "você" in expanded
    assert "hoje" in expanded
    assert "professor" in expanded


def test_protected_negation_in_stopwords():
    pipe = NLPPipeline()
    tokens = ["eu", "não", "entendi", "a", "matéria"]
    filtered = pipe.remove_stopwords(tokens)
    assert "não" in filtered
    assert "entendi" in filtered
    assert "matéria" in filtered
    assert "eu" not in filtered
    assert "a" not in filtered


def test_debug_inspection_structure():
    pipe = NLPPipeline()
    info = pipe.inspect("O que é TF-IDF?")
    assert "1_texto_original" in info
    assert "2_normalizado" in info
    assert "3_expansao_girias" in info
    assert "4_limpeza_regex" in info
    assert "5_tokens_brutos" in info
    assert "6_apos_stopwords" in info
    assert "7_radicais_stemming" in info
    assert "8_texto_final_vetorizador" in info
