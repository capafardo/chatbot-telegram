"""Pipeline Educacional de Processamento de Linguagem Natural (PLN).

Referência Didática:
    - Aulas 02 e 03 do Plano de Ensino (CETAM / IBC).
    - Pré-processamento textual, normalização, limpeza de ruídos, tratamento de gírias,
      tokenização, remoção seletiva de stopwords e radicalização (Stemming).
    - Fornece o raio-X completo das transformações textuais para o modo /debug.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Set
from pathlib import Path

from src.config import SLANG_FILE

# ==============================================================================
# STOPWORDS EM LÍNGUA PORTUGUESA (COM PRESERVAÇÃO DE NEGAÇÃO E PRONOMES INTERROGATIVOS)
# ==============================================================================
# Decisão Arquitetural Crítica:
# Não removemos termos como "não", "nunca", "qual", "como", "quem", "o que",
# pois são indispensáveis para a identificação da intenção do usuário em chatbots.
DEFAULT_STOPWORDS: Set[str] = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "para", "por", "com", "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas", "isto", "isso", "aquilo",
    "meu", "minha", "meus", "minhas", "teu", "tua", "teus", "tuas",
    "seu", "sua", "seus", "suas", "nosso", "nossa", "nossos", "nossas",
    "deles", "delas", "dele", "dela",
    "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas",
    "sou", "somos", "sao", "era", "eramos", "eram", "foi", "fomos", "foram",
    "ser", "sendo", "sido", "estar", "estou", "estamos", "estao", "estava",
    "ter", "tem", "temos", "tem", "tinha", "tinham", "tive", "tivemos"
}

# Palavras protegidas que NUNCA devem ser eliminadas como ruído
PROTECTED_WORDS: Set[str] = {
    "nao", "não", "nunca", "jamais", "sem", "nada", "nenhum", "nenhuma",
    "quem", "qual", "quais", "quando", "onde", "como", "porque", "por que"
}


class NLPPipeline:
    """Pipeline explicável de PLN para assistentes virtuais de caráter didático."""

    def __init__(self, slang_path: Path | None = None) -> None:
        self.slang_dict: Dict[str, str] = {}
        self._load_slang_dict(slang_path or SLANG_FILE)

    def _load_slang_dict(self, path: Path) -> None:
        """Carrega o dicionário externo de gírias e abreviações em JSON."""
        if path.exists():
            try:
                with open(path, mode="r", encoding="utf-8") as f:
                    self.slang_dict = json.load(f)
            except Exception:
                self.slang_dict = {}

    @staticmethod
    def strip_accents(text: str) -> str:
        """Remove marcas de acentuação gráfica preservando o caractere base."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def normalize(self, text: str) -> str:
        """Etapa 1: Converte o texto para minúsculas e remove espaços excedentes."""
        return text.lower().strip()

    def expand_slang(self, text: str) -> str:
        """Etapa 2: Substitui gírias e abreviações de internet por seus termos canônicos."""
        if not self.slang_dict:
            return text

        tokens = text.split()
        expanded_tokens: List[str] = []
        for token in tokens:
            # Verifica pontuação no final do token (ex: 'pq?' -> 'por que?')
            clean_token = re.sub(r"[^\w\s]", "", token)
            if clean_token in self.slang_dict:
                expanded_tokens.append(self.slang_dict[clean_token])
            elif token in self.slang_dict:
                expanded_tokens.append(self.slang_dict[token])
            else:
                expanded_tokens.append(token)
        return " ".join(expanded_tokens)

    @staticmethod
    def clean_noise(text: str) -> str:
        """Etapa 3: Remove caracteres especiais e ruídos gráficos via Expressões Regulares."""
        # Mantém apenas letras, números e espaços
        cleaned = re.sub(r"[^\w\s-]", " ", text)
        # Compacta múltiplos espaços em um único espaço
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Etapa 4: Separa o texto em unidades léxicas individuais (Tokens)."""
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Etapa 5: Filtra palavras de ligação sem semântica primária, preservando negações."""
        filtered: List[str] = []
        for token in tokens:
            normalized_token = self.strip_accents(token)
            if normalized_token in PROTECTED_WORDS or token in PROTECTED_WORDS:
                filtered.append(token)
            elif normalized_token not in DEFAULT_STOPWORDS and token not in DEFAULT_STOPWORDS:
                filtered.append(token)
        return filtered

    def stem_word(self, word: str) -> str:
        """Etapa 6 (Didática): Redução simplificada ao radical (Stemming em Língua Portuguesa).

        Aplica regras heurísticas simples de sufixação para fins pedagógicos
        (mostrando como palavras da mesma raiz convergem para um radical comum).
        """
        word_clean = self.strip_accents(word.lower())
        if len(word_clean) <= 3:
            return word_clean

        # Remoção de plurais simples
        if word_clean.endswith("s") and not word_clean.endswith("is"):
            word_clean = word_clean[:-1]

        # Remoção de sufixos verbais comuns (-ando, -endo, -indo, -aram, -erem, -acao, -coes)
        suffixes = (
            "amento", "imento", "izacao", "adores", "adoras",
            "ando", "endo", "indo", "aram", "erem", "iram",
            "asse", "esse", "isse", "aria", "eria", "iria",
            "ador", "adora", "acao", "coes", "ismo", "ista",
            "mente", "avel", "ivel"
        )
        for suf in suffixes:
            if word_clean.endswith(suf) and len(word_clean) - len(suf) >= 3:
                return word_clean[:-len(suf)]

        # Desinências verbais simples (-ar, -er, -ir)
        if word_clean.endswith(("ar", "er", "ir")) and len(word_clean) >= 5:
            return word_clean[:-2]

        return word_clean

    def stem(self, tokens: List[str]) -> List[str]:
        """Aplica o processo de radicalização a uma lista de tokens."""
        return [self.stem_word(t) for t in tokens]

    def process(self, text: str) -> str:
        """Executa o pipeline completo e retorna a string processada pronta para o TF-IDF."""
        step1 = self.normalize(text)
        step2 = self.expand_slang(step1)
        step3 = self.clean_noise(step2)
        tokens = self.tokenize(step3)
        tokens_filtered = self.remove_stopwords(tokens)
        return " ".join(tokens_filtered)

    def inspect(self, text: str) -> Dict[str, Any]:
        """Gera o raio-X detalhado de cada fase do pipeline para o comando /debug."""
        norm = self.normalize(text)
        slang = self.expand_slang(norm)
        cleaned = self.clean_noise(slang)
        tokens_raw = self.tokenize(cleaned)
        tokens_no_stop = self.remove_stopwords(tokens_raw)
        stems = self.stem(tokens_no_stop)

        return {
            "1_texto_original": text,
            "2_normalizado": norm,
            "3_expansao_girias": slang,
            "4_limpeza_regex": cleaned,
            "5_tokens_brutos": tokens_raw,
            "6_apos_stopwords": tokens_no_stop,
            "7_radicais_stemming": stems,
            "8_texto_final_vetorizador": " ".join(tokens_no_stop),
        }
