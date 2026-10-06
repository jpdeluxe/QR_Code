"""Modelagem preditiva: TF-IDF + LR / SVM linear / Naive Bayes."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

SEED = 42


def vetorizador():
    # n-gramas de 1 e 2 palavras; min_df=5 gera ~12,6 mil atributos,
    # mesma ordem de grandeza dos 13.916 relatados no TCC.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5)


def classificadores():
    return {
        "Regressão Logística": LogisticRegression(max_iter=2000, random_state=SEED),
        "SVM Linear": LinearSVC(random_state=SEED),
        "Naive Bayes": MultinomialNB(),
    }


def escore(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def metricas(y, y_pred, s=None):
    m = {
        "acuracia": accuracy_score(y, y_pred),
        "precisao": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y, y_pred, average="macro"),
        "f1": f1_score(y, y_pred, average="macro"),
    }
    if s is not None:
        m["auc"] = roc_auc_score(y, s)
    return {k: round(float(v), 3) for k, v in m.items()}


def particao(df, modo):
    """Retorna (treino, teste).

    - 'aleatoria': estratificada 80/20, como no TCC (com duplicatas).
    - 'dedup': remove ementas duplicadas antes da partição aleatória.
    - 'temporal': sem duplicatas; 80% mais antigos treinam, 20% mais recentes testam.
    """
    if modo == "aleatoria":
        return train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SEED)
    base = df.drop_duplicates("ementa_text").reset_index(drop=True)
    if modo == "dedup":
        return train_test_split(base, test_size=0.2, stratify=base["y"], random_state=SEED)
    if modo == "temporal":
        base = base.sort_values("data", kind="stable").reset_index(drop=True)
        corte = int(len(base) * 0.8)
        return base.iloc[:corte], base.iloc[corte:]
    raise ValueError(modo)


def avaliar_texto(treino, teste, coluna, nome_clf="SVM Linear", cv=False):
    clf = classificadores()[nome_clf]
    pipe = make_pipeline(vetorizador(), clf)
    pipe.fit(treino[coluna], treino["y"])
    y_pred = pipe.predict(teste[coluna])
    res = metricas(teste["y"], y_pred, escore(pipe, teste[coluna]))
    res["n_atributos"] = len(pipe[0].vocabulary_)
    if cv:
        f1s = cross_val_score(
            make_pipeline(vetorizador(), classificadores()[nome_clf]),
            treino[coluna], treino["y"], scoring="f1_macro",
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        )
        res["cv_f1_media"] = round(float(f1s.mean()), 3)
        res["cv_f1_desvio"] = round(float(f1s.std()), 3)
    return res, pipe, y_pred


def avaliar_metadados(treino, teste, colunas=("orgao_julgador", "judge_relator")):
    """Modelo apenas com atributos conhecidos ANTES do julgamento."""
    pipe = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    cols = list(colunas)
    pipe.fit(treino[cols], treino["y"])
    y_pred = pipe.predict(teste[cols])
    return metricas(teste["y"], y_pred, pipe.predict_proba(teste[cols])[:, 1])


def matriz_confusao(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)}


def ece(y, p, bins=10):
    """Expected Calibration Error."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(
        abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
        for b in range(bins) if (idx == b).any()
    ))


def calibracao(treino, teste, coluna):
    """Compara LR bruta vs. calibrada (isotônica, CV interna) em Brier e ECE."""
    bruta = make_pipeline(vetorizador(), LogisticRegression(max_iter=2000, random_state=SEED))
    bruta.fit(treino[coluna], treino["y"])
    p_bruta = bruta.predict_proba(teste[coluna])[:, 1]

    cal = make_pipeline(
        vetorizador(),
        CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, random_state=SEED), method="isotonic", cv=5
        ),
    )
    cal.fit(treino[coluna], treino["y"])
    p_cal = cal.predict_proba(teste[coluna])[:, 1]
    y = teste["y"].to_numpy()
    return {
        "brier_bruta": round(float(brier_score_loss(y, p_bruta)), 4),
        "brier_calibrada": round(float(brier_score_loss(y, p_cal)), 4),
        "ece_bruta": round(ece(y, p_bruta), 4),
        "ece_calibrada": round(ece(y, p_cal), 4),
    }, p_bruta, p_cal
