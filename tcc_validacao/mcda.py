"""AHP (Saaty, 1980) e TOPSIS (Hwang; Yoon, 1981)."""
import numpy as np

# Índice de consistência aleatório de Saaty
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def ahp(matriz):
    """Pesos pelo autovetor principal + razão de consistência (RC)."""
    A = np.asarray(matriz, dtype=float)
    n = A.shape[0]
    autovalores, autovetores = np.linalg.eig(A)
    i = int(np.argmax(autovalores.real))
    w = np.abs(autovetores[:, i].real)
    w /= w.sum()
    lambda_max = float(autovalores.real[i])
    ic = (lambda_max - n) / (n - 1)
    rc = ic / RI[n] if RI[n] else 0.0
    return w, lambda_max, rc


def topsis(decisao, pesos, beneficio):
    """Coeficiente de proximidade (0 a 1) de cada alternativa.

    decisao: matriz alternativas x critérios
    beneficio: lista de bool (True = quanto maior melhor; False = custo)
    """
    M = np.asarray(decisao, dtype=float)
    N = M / np.sqrt((M ** 2).sum(axis=0))
    V = N * np.asarray(pesos)
    b = np.asarray(beneficio)
    ideal = np.where(b, V.max(0), V.min(0))
    anti = np.where(b, V.min(0), V.max(0))
    d_mais = np.sqrt(((V - ideal) ** 2).sum(1))
    d_menos = np.sqrt(((V - anti) ** 2).sum(1))
    return d_menos / (d_mais + d_menos)


def sensibilidade_pesos(decisao, beneficio, alternativas, idx_prob, passos=101):
    """Varia o peso da probabilidade de êxito de 0 a 1, repartindo o restante
    entre custo e tempo na proporção original do AHP (0,648 : 0,230).

    Retorna lista de (peso_prob, ranking) e o primeiro peso em que cada
    alternativa passa a ser a 1ª colocada.
    """
    custo_tempo = np.array([0.648, 0.230]) / (0.648 + 0.230)
    curvas, lideres = [], {}
    for wp in np.linspace(0, 1, passos):
        w = np.zeros(3)
        outros = [j for j in range(3) if j != idx_prob]
        w[outros] = (1 - wp) * custo_tempo
        w[idx_prob] = wp
        cc = topsis(decisao, w, beneficio)
        curvas.append((round(float(wp), 3), cc))
        lider = alternativas[int(np.argmax(cc))]
        lideres.setdefault(lider, round(float(wp), 3))
    return curvas, lideres
