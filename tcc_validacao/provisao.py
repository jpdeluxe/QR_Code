"""Provisionamento de contingências: valor esperado vs. regra binária."""
import numpy as np
import pandas as pd


def tabela_provisao(p_perda, valor):
    p_perda, valor = np.asarray(p_perda, float), np.asarray(valor, float)
    df = pd.DataFrame({
        "p_perda": p_perda,
        "valor_risco": valor,
        "provisao_esperada": p_perda * valor,
        "provisao_binaria": np.where(p_perda > 0.5, valor, 0.0),
    })
    df.index = [f"Caso {i + 1}" for i in range(len(df))]
    df.loc["Total"] = [np.nan, valor.sum(), df["provisao_esperada"].sum(), df["provisao_binaria"].sum()]
    return df


def backtest_carteira(y_perda, p_perda, valor):
    """Compara a provisão feita com a perda que de fato ocorreu.

    y_perda: 1 se o escritório (recorrente) perdeu = recurso improvido.
    Retorna erro absoluto de provisão e erro por caso para cada regra.
    """
    y, p, v = map(lambda a: np.asarray(a, float), (y_perda, p_perda, valor))
    perda_real = (y * v).sum()
    res = {"perda_real": perda_real}
    for nome, prov in {
        "valor_esperado": p * v,
        "binaria": np.where(p > 0.5, v, 0.0),
    }.items():
        res[f"provisao_{nome}"] = prov.sum()
        res[f"erro_total_{nome}_%"] = 100 * (prov.sum() - perda_real) / perda_real
        res[f"erro_medio_caso_{nome}"] = np.abs(prov - y * v).mean()
    return {k: round(float(x), 2) for k, x in res.items()}
