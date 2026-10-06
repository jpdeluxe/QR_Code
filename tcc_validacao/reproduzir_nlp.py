"""Reproduz as métricas de NLP (Quadros 1 e 2 e calibração).

Uso:  pip install -r requirements.txt  &&  python reproduzir_nlp.py
O corpus (Lage-Freitas et al., 2022) é baixado automaticamente.
"""
import dados
import modelos

df = dados.carregar_corpus()
sw = dados.carregar_stopwords()
textos = {
    "Ementa integral": lambda t: dados.normalizar(t, sw),
    "Ementa sem léxico dispositivo": lambda t: dados.normalizar(t, sw, dados.LEXICO_BASICO),
    "Sem léxico dispositivo ampliado": lambda t: dados.normalizar(t, sw, dados.LEXICO_AMPLIADO),
    "Porção temática inicial (40%)": lambda t: dados.normalizar(
        dados.porcao_inicial(t, 0.40), sw, dados.LEXICO_BASICO),
    "Porção temática inicial (25%)": lambda t: dados.normalizar(
        dados.porcao_inicial(t, 0.25), sw, dados.LEXICO_BASICO),
}
for nome, f in textos.items():
    df[nome] = df["ementa_text"].map(f)
print(f"Registros: {len(df)} | ementas distintas: {df['ementa_text'].nunique()}")

# Quadro 1 — classificadores, sem duplicatas
tr, te = modelos.particao(df, "dedup")
print(f"\nQuadro 1 (treino={len(tr)}, teste={len(te)})")
for clf in modelos.classificadores():
    res, _, _ = modelos.avaliar_texto(tr, te, "Ementa sem léxico dispositivo", clf, cv=True)
    print(f"  {clf:20s} {res}")

# Quadro 2 — F1 (SVM linear) por recorte do texto e tipo de partição
particoes = {m: modelos.particao(df, m) for m in ("aleatoria", "dedup", "temporal")}
print("\nQuadro 2 — F1 macro: aleatória com duplicatas | sem duplicatas | temporal")
for nome in textos:
    f1 = [modelos.avaliar_texto(a, b, nome)[0]["f1"] for a, b in particoes.values()]
    print(f"  {nome:34s} " + " | ".join(f"{v:.3f}" for v in f1))
f1 = [modelos.avaliar_metadados(a, b)["f1"] for a, b in particoes.values()]
print(f"  {'Somente metadados (órgão e relator)':34s} " + " | ".join(f"{v:.3f}" for v in f1))

# Calibração — ajuste (LR) e validação (isotônica) disjuntos do teste temporal
tr_t, te_t = particoes["temporal"]
cal, _, _ = modelos.calibracao(tr_t, te_t, "Ementa sem léxico dispositivo")
print("\nCalibração (teste temporal):", cal)
