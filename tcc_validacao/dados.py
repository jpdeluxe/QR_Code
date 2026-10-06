"""Carga e preparação do corpus TJAL (Lage-Freitas et al., 2022).

Recorte: acórdãos de órgãos cíveis com desfecho binário
(provido = 1 / improvido = 0), totalizando 5.235 registros.
"""
import io
import re
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL_DATASET = (
    "https://raw.githubusercontent.com/proflage/"
    "predicting-brazilian-court-decisions/main/dataset.zip"
)
PASTA_DADOS = Path(__file__).parent / "dados"
CSV_LOCAL = PASTA_DADOS / "dataset.csv"

# Léxico dispositivo básico: termos que explicitam o veredicto
LEXICO_BASICO = [
    "provido", "provida", "providos", "providas", "improvido", "improvida",
    "improvidos", "desprovido", "desprovida", "provimento", "desprovimento",
    "improvimento", "negado", "negada", "negados", "nego", "mantido",
    "mantida", "mantidos", "reformada", "reformado", "procedente",
    "improcedente", "procedencia", "improcedencia", "parcialmente",
]
# Léxico ampliado: acrescenta marcadores dispositivos que sobreviveram ao
# léxico básico (acolhidos, rejeitados, manutenção, extinção...) e termos de
# quórum/dispositivo do acórdão.
LEXICO_AMPLIADO = LEXICO_BASICO + [
    "acolhido", "acolhida", "acolhidos", "acolhidas", "rejeitado",
    "rejeitada", "rejeitados", "rejeitadas", "conhecido", "conhecida",
    "conhecidos", "conhecidas", "unanimidade", "unanime", "maioria",
    "votos", "manutencao", "mantem", "mantenho", "reforma", "reformar",
    "nega", "negar", "negou", "dou", "dar", "deu", "dado", "recurso",
    "recursos", "apelo", "sentenca", "decisao", "infringentes",
    "retorno", "autos", "inversao", "sucumbencia", "extincao", "nulidade",
    "anulada", "anulacao", "cassada", "prejudicado", "prejudicada",
]


def baixar_dataset() -> Path:
    """Baixa o dataset público (uma vez) para tcc_validacao/dados/."""
    if CSV_LOCAL.exists():
        return CSV_LOCAL
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL_DATASET, timeout=60) as resp:
        conteudo = resp.read()
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        CSV_LOCAL.write_bytes(zf.read("dataset.csv"))
    return CSV_LOCAL


def carregar_corpus() -> pd.DataFrame:
    """Retorna o corpus binário de órgãos cíveis (5.235 linhas).

    Observação: em ~18 linhas da 'Seção Especializada Cível' o CSV vem com as
    colunas deslocadas; o rótulo correto está em 'decision_label' mesmo assim.
    """
    df = pd.read_csv(baixar_dataset(), sep="<=>", engine="python")
    df = df[df["orgao_julgador"].str.contains("Cível", na=False)]
    df = df[df["decision_label"].isin(["yes", "no"])].copy()
    df["y"] = (df["decision_label"] == "yes").astype(int)
    df["data"] = pd.to_datetime(df["publish_date"], format="%d/%m/%Y", errors="coerce")
    df["ementa_text"] = df["ementa_text"].astype(str)
    return df.reset_index(drop=True)


def _sem_acento(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


def carregar_stopwords() -> set:
    linhas = (PASTA_DADOS / "stopwords_pt.txt").read_text(encoding="utf-8").split()
    return {_sem_acento(p) for p in linhas}


def normalizar(texto: str, stopwords: set, lexico: list | None = None) -> str:
    """Caixa baixa, sem acento, sem pontuação/números, sem stopwords e,
    opcionalmente, sem o léxico dispositivo."""
    texto = _sem_acento(texto.lower())
    texto = re.sub(r"[^a-z\s]", " ", texto)
    remover = stopwords | set(lexico or [])
    return " ".join(t for t in texto.split() if t not in remover and len(t) > 1)


def porcao_inicial(texto: str, fracao: float) -> str:
    """Mantém apenas a fração inicial (em palavras) da ementa."""
    palavras = texto.split()
    return " ".join(palavras[: max(1, int(len(palavras) * fracao))])
