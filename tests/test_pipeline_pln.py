"""Testes unitários para o PipelinePLN."""

from src.pipeline_pln import PipelinePLN


def test_expansao_girias():
    pipeline = PipelinePLN()
    texto = "vc vai na aula hj prof?"
    expandido = pipeline.expandir_girias(texto.lower())
    assert "você" in expandido
    assert "hoje" in expandido
    assert "professor" in expandido


def test_preservacao_negacao_e_interrogativos_nas_stopwords():
    pipeline = PipelinePLN()
    tokens = ["eu", "não", "entendi", "a", "matéria"]
    filtrados = pipeline.remover_stopwords(tokens)
    assert "não" in filtrados
    assert "entendi" in filtrados
    assert "matéria" in filtrados
    assert "eu" not in filtrados
    assert "a" not in filtrados


def test_preservacao_por_que_apos_giria_pq():
    pipeline = PipelinePLN()
    processado = pipeline.processar("pq não podemos remover stopwords?")
    tokens_finais = processado.split()
    assert "por" in tokens_finais
    assert "que" in tokens_finais
    assert "nao" in tokens_finais


def test_normalizacao_acentos_para_vetorizador():
    pipeline = PipelinePLN(usar_stemming=False)
    com_acento = pipeline.processar("Como funciona a avaliação de regressão logística?")
    sem_acento = pipeline.processar("Como funciona a avaliacao de regressao logistica?")
    assert com_acento == sem_acento


def test_estrutura_inspecao_debug():
    pipeline = PipelinePLN()
    info = pipeline.inspecionar("O que é TF-IDF?")
    assert "1_texto_original" in info
    assert "2_normalizado" in info
    assert "3_expansao_girias" in info
    assert "4_limpeza_regex" in info
    assert "5_tokens_brutos" in info
    assert "6_apos_stopwords" in info
    assert "7_radicais_stemming" in info
    assert "7_stemming_ativo_no_modelo" in info
    assert "8_texto_final_vetorizador" in info
