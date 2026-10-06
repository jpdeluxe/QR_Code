"""Gera as figuras e números usados na versão revisada do TCC.

Cenário principal: corpus sem ementas duplicadas (2.324 acórdãos).
Calibração, distribuição de probabilidades e provisão: partição temporal.

Uso:  python figuras_tcc.py   ->  resultados/tcc/*.png e numeros_tcc.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

import dados
import mcda
import modelos
import provisao

AZUL, LARANJA, VERDE = "#2c5985", "#b5651d", "#4a8a3c"
SAIDA = Path(__file__).parent / "resultados" / "tcc"
SAIDA.mkdir(parents=True, exist_ok=True)
N = {}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


def salvar(nome):
    plt.tight_layout()
    plt.savefig(SAIDA / nome, dpi=150)
    plt.close()


df = dados.carregar_corpus()
sw = dados.carregar_stopwords()
df["txt"] = df["ementa_text"].map(lambda t: dados.normalizar(t, sw, dados.LEXICO_TCC))
df["txt_integral"] = df["ementa_text"].map(lambda t: dados.normalizar(t, sw))
df["txt_amp"] = df["ementa_text"].map(lambda t: dados.normalizar(t, sw, dados.LEXICO_AMPLIADO))
for f in (0.40, 0.25):
    df[f"txt_{int(f * 100)}"] = df["ementa_text"].map(
        lambda t: dados.normalizar(dados.porcao_inicial(t, f), sw, dados.LEXICO_TCC))
base = df.drop_duplicates("ementa_text")
N["n_total"], N["n_unicas"] = len(df), len(base)

# Figura 1 — taxa por câmara (sem duplicatas)
cam = base[base["orgao_julgador"].str.contains("Câmara")]
g = cam.groupby("orgao_julgador")["y"].agg(["mean", "size"])
N["camaras"] = {k: [round(float(m), 3), int(n)] for k, (m, n) in g.iterrows()}
plt.figure(figsize=(6.6, 3.8))
rot = [k.replace(" Câmara Cível", " C. Cív.") for k in g.index]
plt.barh(rot, g["mean"] * 100, color=AZUL)
for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
    plt.text(m * 100 + 0.5, i, f"{m * 100:.1f}% (n={n})", va="center", fontsize=8)
plt.xlim(0, 50); plt.xlabel("Taxa de provimento (%)")
plt.title("Taxa de provimento por câmara cível (TJAL)")
salvar("fig1_camaras.png")

# Quadro 1 / Figura 2 — classificadores (sem duplicatas)
tr, te = modelos.particao(df, "dedup")
N["n_treino"], N["n_teste"] = len(tr), len(te)
N["quadro1"] = {}
for nome in modelos.classificadores():
    res, pipe, y_pred = modelos.avaliar_texto(tr, te, "txt", nome, cv=True)
    N["quadro1"][nome] = res
    if nome == "SVM Linear":
        svm, y_svm = pipe, y_pred
    if nome == "Regressão Logística":
        lr = pipe
N["n_atributos"] = N["quadro1"]["SVM Linear"]["n_atributos"]
plt.figure(figsize=(6.6, 3.9))
x = np.arange(3)
acc = [N["quadro1"][k]["acuracia"] for k in N["quadro1"]]
f1 = [N["quadro1"][k]["f1"] for k in N["quadro1"]]
for dx, vals, cor, lab in ((-0.19, acc, AZUL, "Acurácia"), (0.19, f1, LARANJA, "F1-score")):
    plt.bar(x + dx, vals, 0.38, color=cor, label=lab)
    for xi, v in zip(x + dx, vals):
        plt.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
plt.xticks(x, ["Regressão\nLogística", "SVM\nLinear", "Naive\nBayes"])
plt.ylim(0, 1.08); plt.ylabel("Desempenho"); plt.legend(fontsize=8)
plt.title("Desempenho comparativo dos classificadores")
salvar("fig2_modelos.png")

# Figura 3 — ROC SVM
s = svm.decision_function(te["txt"])
fpr, tpr, _ = roc_curve(te["y"], s)
auc = N["quadro1"]["SVM Linear"]["auc"]
plt.figure(figsize=(5.4, 4.0))
plt.plot(fpr, tpr, color=AZUL, lw=2, label=f"SVM linear (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
plt.xlabel("Taxa de falsos positivos"); plt.ylabel("Taxa de verdadeiros positivos")
plt.title("Curva ROC"); plt.legend(loc="lower right", fontsize=8)
salvar("fig3_roc.png")

# Figura 4 — matriz de confusão
cm = modelos.matriz_confusao(te["y"], y_svm)
N["confusao"] = cm
M = np.array([[cm["VN"], cm["FP"]], [cm["FN"], cm["VP"]]])
plt.figure(figsize=(4.6, 3.9)); plt.grid(False)
plt.imshow(M, cmap="Blues"); plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, M[i, j], ha="center", va="center", fontsize=14, fontweight="bold",
                 color="white" if M[i, j] > M.max() / 2 else "black")
plt.xticks([0, 1], ["Improvido", "Provido"]); plt.yticks([0, 1], ["Improvido", "Provido"])
plt.xlabel("Predito"); plt.ylabel("Observado"); plt.title("Matriz de confusão — SVM linear")
salvar("fig4_confusao.png")

# Quadro 2 — sensibilidade (sem duplicatas e temporal)
tr_t, te_t = modelos.particao(df, "temporal")
N["temporal"] = {"n_treino": len(tr_t), "n_teste": len(te_t),
                 "teste_inicio": str(te_t["data"].min().date()),
                 "teste_fim": str(te_t["data"].max().date())}
N["quadro2"] = {}
for rot, col in (("Ementa integral", "txt_integral"), ("Sem léxico (TCC)", "txt"),
                 ("Sem léxico ampliado", "txt_amp"), ("40%", "txt_40"), ("25%", "txt_25")):
    N["quadro2"][rot] = {
        "aleatoria_com_dup": modelos.avaliar_texto(*modelos.particao(df, "aleatoria"), col)[0],
        "dedup": modelos.avaliar_texto(tr, te, col)[0],
        "temporal": modelos.avaliar_texto(tr_t, te_t, col)[0],
    }
N["metadados"] = {"dedup": modelos.avaliar_metadados(tr, te),
                  "temporal": modelos.avaliar_metadados(tr_t, te_t)}
N["gemeos_teste_%"] = round(100 * modelos.particao(df, "aleatoria")[1]["ementa_text"].isin(
    set(modelos.particao(df, "aleatoria")[0]["ementa_text"])).mean(), 1)

# Quadro 4 — termos (LR, sem duplicatas)
vocab = np.array(lr[0].get_feature_names_out()); coef = lr[-1].coef_[0]
N["quadro4"] = {"provimento": vocab[np.argsort(coef)[-10:][::-1]].tolist(),
                "improvimento": vocab[np.argsort(coef)[:10]].tolist()}

# AHP + TOPSIS + sensibilidade (Figura 5)
w, _, rc = mcda.ahp([[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]])
decisao = np.array([[2.0, 3.0, 0.20], [12.0, 12.0, 0.60], [30.0, 34.0, 0.78]])
alts = ["Não recorrer", "Propor acordo", "Prosseguir no litígio"]
cc = mcda.topsis(decisao, w, [False, False, True])
curvas, lideres = mcda.sensibilidade_pesos(decisao, [False, False, True], alts, 2)
N["ahp"] = {"pesos": np.round(w, 3).tolist(), "rc": round(rc, 4)}
N["topsis"] = dict(zip(alts, np.round(cc, 3).tolist()))
N["lideres"] = lideres
plt.figure(figsize=(6.6, 3.9))
for j, (a, cor) in enumerate(zip(alts, (AZUL, LARANJA, VERDE))):
    plt.plot([c[0] for c in curvas], [c[1][j] for c in curvas], color=cor, lw=2, label=a)
plt.axvline(w[2], ls="--", c="gray", lw=1)
plt.text(w[2] + 0.01, 0.03, "peso atual\n(0,122)", fontsize=8)
plt.xlabel("Peso da probabilidade de êxito"); plt.ylabel("Coeficiente de proximidade")
plt.title("Sensibilidade do TOPSIS ao peso da probabilidade de êxito")
plt.legend(fontsize=8); salvar("fig5_sensibilidade_pesos.png")

# Calibração (temporal) — Figuras 6 e 8
cal, p_bruta, p_cal = modelos.calibracao(tr_t, te_t, "txt")
N["calibracao"] = cal
perda = 1 - p_cal
plt.figure(figsize=(6.6, 3.9))
plt.hist(perda, bins=20, range=(0, 1), color=AZUL, edgecolor="white")
plt.axvline(0.5, ls="--", color=LARANJA, label="Limiar de 50% (provável)")
plt.xlabel("Probabilidade estimada de perda (calibrada)"); plt.ylabel("Nº de processos")
plt.title("Distribuição das probabilidades de perda (teste temporal)")
plt.legend(fontsize=8); salvar("fig6_probdist.png")
N["perda_acima_50_%"] = round(100 * float((perda > 0.5).mean()), 1)

yt = te_t["y"].to_numpy()
plt.figure(figsize=(5.4, 4.2))
for nome, p, cor in (("Regressão Logística (bruta)", p_bruta, LARANJA),
                     ("Calibrada (isotônica)", p_cal, AZUL)):
    idx = np.minimum((p * 10).astype(int), 9)
    pts = [(p[idx == b].mean(), yt[idx == b].mean()) for b in range(10) if (idx == b).any()]
    plt.plot(*zip(*pts), "o-", color=cor, label=nome)
plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Calibração perfeita")
plt.xlabel("Probabilidade prevista de provimento"); plt.ylabel("Frequência observada")
plt.title("Curva de calibração (teste temporal)"); plt.legend(fontsize=8)
salvar("fig8_calibracao.png")

# Quadro 6 — 5 processos reais do teste temporal (probabilidade calibrada)
alvos = [0.05, 0.35, 0.75, 0.90, 0.97]
casos = [int(np.argmin(np.abs(perda - a))) for a in alvos]
p6 = np.round(perda[casos], 3)
tab = provisao.tabela_provisao(p6, [120, 80, 250, 45, 300])
N["quadro6"] = tab.round({"p_perda": 3, "valor_risco": 1, "provisao_esperada": 1, "provisao_binaria": 1}).reset_index().to_dict(orient="records")
N["quadro6_desfecho_real"] = ["perda" if yt[c] == 0 else "êxito" for c in casos]

rng = np.random.default_rng(modelos.SEED)
valores = rng.uniform(20, 300, len(te_t))
N["backtest"] = {
    "bruta": provisao.backtest_carteira(1 - yt, 1 - p_bruta, valores),
    "calibrada": provisao.backtest_carteira(1 - yt, perda, valores),
}

(SAIDA / "numeros_tcc.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=float))
print(json.dumps(N, indent=1, ensure_ascii=False, default=float))
