"""Pipeline Educacional de Processamento de Linguagem Natural (PLN).

Referência Didática:
    - Aulas 02 e 03 do Plano de Ensino (CETAM / IBC).
    - Pré-processamento textual, normalização, limpeza de ruídos, tratamento de gírias,
      tokenização, remoção seletiva de stopwords e radicalização (Stemming).
    - Fornece o raio-X completo das transformações textuais para o modo /debug.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set

from src.configuracao import ARQUIVO_GIRIAS, USAR_STEMMING

logger = logging.getLogger(__name__)

# ==============================================================================
# STOPWORDS EM LÍNGUA PORTUGUESA (COM PRESERVAÇÃO DE NEGAÇÃO E PRONOMES INTERROGATIVOS)
# ==============================================================================
# Decisão Arquitetural Crítica:
# Não removemos termos como "não", "nunca", "qual", "como", "quem", "por", "que",
# pois são indispensáveis para a identificação de perguntas e negações em chatbots.
STOPWORDS_PADRAO: Set[str] = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "para", "com", "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas", "isto", "isso", "aquilo",
    "meu", "minha", "meus", "minhas", "teu", "tua", "teus", "tuas",
    "seu", "sua", "seus", "suas", "nosso", "nossa", "nossos", "nossas",
    "deles", "delas", "dele", "dela",
    "eu", "tu", "ele", "ela", "vos", "eles", "elas",
    "sou", "somos", "sao", "era", "eramos", "eram", "foi", "fomos", "foram",
    "ser", "sendo", "sido", "estar", "estou", "estamos", "estao", "estava",
    "ter", "tem", "temos", "tinha", "tinham", "tive", "tivemos"
}

# Palavras protegidas (em tokens individuais) que NUNCA devem ser eliminadas como ruído
PALAVRAS_PROTEGIDAS: Set[str] = {
    "nao", "não", "nunca", "jamais", "sem", "nada", "nenhum", "nenhuma", "nem",
    "quem", "qual", "quais", "quando", "onde", "como", "porque", "por", "que"
}


class PipelinePLN:
    """Pipeline explicável de PLN para assistentes virtuais de caráter didático."""

    def __init__(
        self,
        caminho_girias: Path | None = None,
        usar_stemming: bool = USAR_STEMMING
    ) -> None:
        self.dicionario_girias: Dict[str, str] = {}
        self.usar_stemming: bool = usar_stemming
        self._carregar_dicionario_girias(caminho_girias or ARQUIVO_GIRIAS)

    def _carregar_dicionario_girias(self, caminho: Path) -> None:
        """Carrega o dicionário externo de gírias e abreviações em JSON."""
        if caminho.exists():
            try:
                with open(caminho, mode="r", encoding="utf-8") as f:
                    self.dicionario_girias = json.load(f)
            except Exception as erro:
                logger.warning(
                    f"Aviso: Não foi possível carregar o dicionário de gírias em '{caminho}': {erro}"
                )
                self.dicionario_girias = {}

    @staticmethod
    def remover_acentos(texto: str) -> str:
        """Remove marcas de acentuação gráfica preservando o caractere base."""
        nfkd = unicodedata.normalize("NFKD", texto)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def normalizar(self, texto: str) -> str:
        """Etapa 1: Converte o texto para minúsculas e remove espaços excedentes."""
        return texto.lower().strip()

    def expandir_girias(self, texto: str) -> str:
        """Etapa 2: Substitui gírias e abreviações de internet por seus termos canônicos."""
        if not self.dicionario_girias:
            return texto

        tokens = texto.split()
        tokens_expandidos: List[str] = []
        for token in tokens:
            # Verifica o token sem pontuação (ex: 'pq?' -> 'pq')
            token_limpo = re.sub(r"[^\w\s]", "", token)
            if token in self.dicionario_girias:
                tokens_expandidos.append(self.dicionario_girias[token])
            elif token_limpo in self.dicionario_girias:
                tokens_expandidos.append(self.dicionario_girias[token_limpo])
            else:
                tokens_expandidos.append(token)
        return " ".join(tokens_expandidos)

    @staticmethod
    def limpar_ruido(texto: str) -> str:
        """Etapa 3: Remove caracteres especiais e ruídos gráficos via Expressões Regulares."""
        # Mantém apenas letras, números, espaços e hífens
        limpo = re.sub(r"[^\w\s-]", " ", texto)
        # Compacta múltiplos espaços em um único espaço
        return re.sub(r"\s+", " ", limpo).strip()

    @staticmethod
    def tokenizar(texto: str) -> List[str]:
        """Etapa 4: Separa o texto em unidades léxicas individuais (Tokens)."""
        return texto.split()

    def remover_stopwords(self, tokens: List[str]) -> List[str]:
        """Etapa 5: Filtra palavras de ligação sem semântica primária, preservando negações."""
        filtrados: List[str] = []
        for token in tokens:
            token_sem_acento = self.remover_acentos(token)
            if token_sem_acento in PALAVRAS_PROTEGIDAS or token in PALAVRAS_PROTEGIDAS:
                filtrados.append(token)
            elif token_sem_acento not in STOPWORDS_PADRAO and token not in STOPWORDS_PADRAO:
                filtrados.append(token)
        return filtrados

    def radicalizar_palavra(self, palavra: str) -> str:
        """Etapa 6 (Didática): Redução simplificada ao radical (Stemming em Língua Portuguesa).

        Aplica regras heurísticas simples de sufixação para fins pedagógicos
        (mostrando como palavras da mesma raiz convergem para um radical comum).
        """
        palavra_limpa = self.remover_acentos(palavra.lower())
        if len(palavra_limpa) <= 3:
            return palavra_limpa

        # Remoção de plurais simples
        if palavra_limpa.endswith("s") and not palavra_limpa.endswith("is"):
            palavra_limpa = palavra_limpa[:-1]

        # Remoção de sufixos verbais e nominais comuns
        sufixos = (
            "amento", "imento", "izacao", "adores", "adoras",
            "ando", "endo", "indo", "aram", "erem", "iram",
            "asse", "esse", "isse", "aria", "eria", "iria",
            "ador", "adora", "acao", "coes", "ismo", "ista",
            "mente", "avel", "ivel"
        )
        for sufixo in sufixos:
            if palavra_limpa.endswith(sufixo) and len(palavra_limpa) - len(sufixo) >= 3:
                return palavra_limpa[:-len(sufixo)]

        # Desinências verbais simples (-ar, -er, -ir)
        if palavra_limpa.endswith(("ar", "er", "ir")) and len(palavra_limpa) >= 5:
            return palavra_limpa[:-2]

        return palavra_limpa

    def radicalizar(self, tokens: List[str]) -> List[str]:
        """Aplica o processo de radicalização a uma lista de tokens."""
        return [self.radicalizar_palavra(t) for t in tokens]

    def processar(self, texto: str) -> str:
        """Executa o pipeline completo e retorna a string processada pronta para o TF-IDF.

        Importante:
            Remove acentos ao final para garantir que palavras digitadas com ou sem
            acento (ex: 'avaliacao' vs 'avaliação') gerem exatamente a mesma representação
            no vetorizador TF-IDF. Se `self.usar_stemming` estiver ativo, utiliza os radicais.
        """
        etapa1 = self.normalizar(texto)
        etapa2 = self.expandir_girias(etapa1)
        etapa3 = self.limpar_ruido(etapa2)
        tokens = self.tokenizar(etapa3)
        tokens_filtrados = self.remover_stopwords(tokens)

        if self.usar_stemming:
            tokens_finais = self.radicalizar(tokens_filtrados)
        else:
            tokens_finais = [self.remover_acentos(t) for t in tokens_filtrados]

        return " ".join(tokens_finais)

    def inspecionar(self, texto: str) -> Dict[str, Any]:
        """Gera o raio-X detalhado de cada fase do pipeline para o comando /debug."""
        normalizado = self.normalizar(texto)
        girias = self.expandir_girias(normalizado)
        limpo = self.limpar_ruido(girias)
        tokens_brutos = self.tokenizar(limpo)
        tokens_sem_stop = self.remover_stopwords(tokens_brutos)
        radicais = self.radicalizar(tokens_sem_stop)

        if self.usar_stemming:
            texto_final = " ".join(radicais)
        else:
            texto_final = " ".join(self.remover_acentos(t) for t in tokens_sem_stop)

        return {
            "1_texto_original": texto,
            "2_normalizado": normalizado,
            "3_expansao_girias": girias,
            "4_limpeza_regex": limpo,
            "5_tokens_brutos": tokens_brutos,
            "6_apos_stopwords": tokens_sem_stop,
            "7_radicais_stemming": radicais,
            "7_stemming_ativo_no_modelo": self.usar_stemming,
            "8_texto_final_vetorizador": texto_final,
        }
