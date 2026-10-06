"""Executa toda a validação do TCC e grava resultados em resultados/.

Uso:  python executar.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dados
import mcda
import modelos
import provisao

SAIDA = Path(__file__).parent / "resultados"
FIG = SAIDA / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
R = {}


def secao(titulo):
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


# ---------------------------------------------------------------- 1. Dados
secao("1. Corpus")
df = dados.carregar_corpus()
sw = dados.carregar_stopwords()
df["txt_integral"] = df["ementa_text"].map(lambda t: dados.normalizar(t, sw))
df["txt_sem_lexico"] = df["ementa_text"].map(lambda t: dados.normalizar(t, sw, dados.LEXICO_BASICO))
df["txt_sem_lexico_ampliado"] = df["ementa_text"].map(
    lambda t: dados.normalizar(t, sw, dados.LEXICO_AMPLIADO))
for f in (0.40, 0.25, 0.10):
    df[f"txt_inicio_{int(f * 100)}"] = df["ementa_text"].map(
        lambda t: dados.normalizar(dados.porcao_inicial(t, f), sw, dados.LEXICO_BASICO))

dup = int(df["ementa_text"].duplicated().sum())
taxa_camara = df.groupby("orgao_julgador")["y"].mean().round(3).to_dict()
R["corpus"] = {
    "n_registros": len(df),
    "n_providos": int(df["y"].sum()),
    "n_improvidos": int((1 - df["y"]).sum()),
    "ementas_duplicadas": dup,
    "ementas_unicas": int(df["ementa_text"].nunique()),
    "periodo": [str(df["data"].min().date()), str(df["data"].max().date())],
    "taxa_provimento_por_orgao": taxa_camara,
}
print(json.dumps(R["corpus"], indent=2, ensure_ascii=False))

# ------------------------------------------- 2. Replicação do Quadro 1 (TCC)
secao("2. Replicação Quadro 1 — partição aleatória (como no TCC)")
treino, teste = modelos.particao(df, "aleatoria")
R["quadro1_replicacao"] = {}
for nome in modelos.classificadores():
    res, pipe, y_pred = modelos.avaliar_texto(
        treino, teste, "txt_sem_lexico", nome, cv=(nome == "SVM Linear"))
    R["quadro1_replicacao"][nome] = res
    if nome == "SVM Linear":
        R["matriz_confusao_svm"] = modelos.matriz_confusao(teste["y"], y_pred)
    print(nome, res)
print("Matriz de confusão SVM:", R["matriz_confusao_svm"])
# Quanto do teste tem "gêmeo" idêntico no treino?
R["teste_com_gemeo_no_treino_%"] = round(
    100 * teste["ementa_text"].isin(set(treino["ementa_text"])).mean(), 1)
print("Teste com ementa idêntica no treino: %.1f%%" % R["teste_com_gemeo_no_treino_%"])

# Termos de maior peso (Quadro 4) — LR
_, pipe_lr, _ = modelos.avaliar_texto(treino, teste, "txt_sem_lexico", "Regressão Logística")
vocab = np.array(pipe_lr[0].get_feature_names_out())
coef = pipe_lr[-1].coef_[0]
R["quadro4_termos"] = {
    "provimento": vocab[np.argsort(coef)[-12:][::-1]].tolist(),
    "improvimento": vocab[np.argsort(coef)[:12]].tolist(),
}
print(json.dumps(R["quadro4_termos"], indent=1, ensure_ascii=False))

# --------------------------------- 3. Sensibilidade (Quadro 2) x 3 partições
secao("3. Sensibilidade do texto x tipo de partição (SVM Linear)")
recortes = {
    "Ementa integral": "txt_integral",
    "Sem léxico dispositivo (TCC)": "txt_sem_lexico",
    "Sem léxico dispositivo ampliado": "txt_sem_lexico_ampliado",
    "Porção inicial 40%": "txt_inicio_40",
    "Porção inicial 25%": "txt_inicio_25",
    "Porção inicial 10%": "txt_inicio_10",
}
R["sensibilidade"] = {}
for modo in ("aleatoria", "dedup", "temporal"):
    tr, te = modelos.particao(df, modo)
    R["sensibilidade"][modo] = {"n_treino": len(tr), "n_teste": len(te)}
    for rotulo, col in recortes.items():
        res, _, _ = modelos.avaliar_texto(tr, te, col, "SVM Linear")
        R["sensibilidade"][modo][rotulo] = res
        print(f"{modo:9s} | {rotulo:33s} | acc={res['acuracia']:.3f} f1={res['f1']:.3f} auc={res['auc']:.3f}")
    meta = modelos.avaliar_metadados(tr, te)
    R["sensibilidade"][modo]["Somente metadados (órgão + relator)"] = meta
    print(f"{modo:9s} | {'Somente metadados (órgão+relator)':33s} | acc={meta['acuracia']:.3f} f1={meta['f1']:.3f} auc={meta['auc']:.3f}")

# Linha de base: classe majoritária no teste temporal
tr_t, te_t = modelos.particao(df, "temporal")
R["baseline_majoritaria_temporal"] = modelos.metricas(te_t["y"], np.zeros(len(te_t), int))
R["periodo_temporal"] = {
    "treino": [str(tr_t["data"].min().date()), str(tr_t["data"].max().date())],
    "teste": [str(te_t["data"].min().date()), str(te_t["data"].max().date())],
}

# ------------------------------------------------------- 4. Calibração
secao("4. Calibração das probabilidades (LR, partição temporal)")
cal, p_bruta, p_cal = modelos.calibracao(tr_t, te_t, "txt_sem_lexico")
R["calibracao_temporal"] = cal
print(cal)

# ------------------------------------------------------------ 5. AHP
secao("5. AHP")
matriz_ahp = [[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]]  # custo, tempo, prob
w, lmax, rc = mcda.ahp(matriz_ahp)
R["ahp"] = {
    "matriz (custo, tempo, prob)": [[round(x, 3) for x in l] for l in matriz_ahp],
    "pesos": dict(zip(["custo", "tempo", "prob_exito"], np.round(w, 3).tolist())),
    "lambda_max": round(lmax, 4), "RC": round(rc, 4),
}
print(R["ahp"])

# ---------------------------------------------------------- 6. TOPSIS
secao("6. TOPSIS + sensibilidade dos pesos")
alternativas = ["Não recorrer", "Propor acordo", "Prosseguir no litígio"]
# Matriz de decisão RECONSTRUÍDA (o TCC não publica os valores): custo em
# R$ mil, tempo em meses, probabilidade de êxito. Reproduz 0,894/0,651/0,106.
decisao = np.array([
    [2.0, 3.0, 0.20],    # Não recorrer
    [12.0, 12.0, 0.60],  # Propor acordo
    [30.0, 34.0, 0.78],  # Prosseguir no litígio (34 meses: CNJ, 2024)
])
beneficio = [False, False, True]
cc = mcda.topsis(decisao, w, beneficio)
R["topsis"] = {
    "matriz_reconstruida": decisao.tolist(),
    "coef_proximidade": dict(zip(alternativas, np.round(cc, 3).tolist())),
}
print(R["topsis"])
curvas, lideres = mcda.sensibilidade_pesos(decisao, beneficio, alternativas, idx_prob=2)
R["topsis_sensibilidade"] = {"peso_prob_em_que_cada_alternativa_lidera": lideres}
print("Peso da prob. de êxito a partir do qual cada alternativa vira 1ª:", lideres)

wp = [c[0] for c in curvas]
plt.figure(figsize=(7, 4))
for j, alt in enumerate(alternativas):
    plt.plot(wp, [c[1][j] for c in curvas], label=alt)
plt.axvline(w[2], ls="--", c="gray", lw=1)
plt.text(w[2] + 0.01, 0.02, "peso TCC\n(0,122)", fontsize=8)
plt.xlabel("Peso da probabilidade de êxito"); plt.ylabel("Coef. de proximidade (TOPSIS)")
plt.title("Sensibilidade do ranking ao peso da probabilidade de êxito")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "sensibilidade_pesos_topsis.png", dpi=150)
plt.close()

# Variando p(êxito) do litígio com os pesos do TCC
ps = np.linspace(0, 1, 101)
cc_p = [mcda.topsis(np.vstack([decisao[:2], [30, 34, p]]), w, beneficio) for p in ps]
R["topsis_prosseguir_lidera_com_pesos_tcc_para_p"] = (
    "nunca" if not any(np.argmax(c) == 2 for c in cc_p)
    else round(float(ps[[np.argmax(c) == 2 for c in cc_p].index(True)]), 2))

# ------------------------------------------------------- 7. Provisionamento
secao("7. Provisionamento")
p6 = [0.004, 0.359, 0.805, 0.927, 0.996]
v6 = [120, 80, 250, 45, 300]
tab = provisao.tabela_provisao(p6, v6).round(2)
R["quadro6_recalculo"] = tab.reset_index().to_dict(orient="records")
print(tab)

# Backtest em carteira real do teste temporal (valores hipotéticos, seed fixa)
rng = np.random.default_rng(modelos.SEED)
valores = rng.uniform(20, 300, len(te_t))  # R$ mil
y_perda = 1 - te_t["y"].to_numpy()
R["backtest_provisao_temporal"] = {
    "LR_bruta": provisao.backtest_carteira(y_perda, 1 - p_bruta, valores),
    "LR_calibrada": provisao.backtest_carteira(y_perda, 1 - p_cal, valores),
}
print(json.dumps(R["backtest_provisao_temporal"], indent=1))

# ----------------------------------------------------------------- Figuras
labels = list(recortes) + ["Somente metadados (órgão + relator)"]
x = np.arange(len(labels))
plt.figure(figsize=(9, 4.5))
for k, modo in enumerate(("aleatoria", "dedup", "temporal")):
    plt.bar(x + (k - 1) * 0.27, [R["sensibilidade"][modo][l]["f1"] for l in labels], 0.27,
            label={"aleatoria": "Aleatória (TCC)", "dedup": "Sem duplicatas",
                   "temporal": "Temporal"}[modo])
plt.xticks(x, [l.replace(" (", "\n(").replace("dispositivo ", "dispositivo\n") for l in labels],
           fontsize=7)
plt.ylabel("F1-score (macro)"); plt.ylim(0.4, 1)
plt.title("F1 do SVM linear por recorte do texto e tipo de partição")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "f1_recorte_particao.png", dpi=150)
plt.close()

plt.figure(figsize=(5, 5))
for nome, p in (("LR bruta", p_bruta), ("LR calibrada", p_cal)):
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins[1:-1])
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [te_t["y"].to_numpy()[idx == b].mean() for b in range(10) if (idx == b).any()]
    plt.plot(xs, ys, "o-", label=nome)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("Probabilidade prevista de provimento"); plt.ylabel("Frequência observada")
plt.title("Curva de calibração (teste temporal)"); plt.legend(); plt.tight_layout()
plt.savefig(FIG / "calibracao.png", dpi=150); plt.close()

(SAIDA / "resultados.json").write_text(
    json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("\nResultados gravados em", SAIDA)
