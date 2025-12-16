# -*- coding: utf-8 -*-
# PASO 1 — CARGA Y COMPRENSIÓN
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CSV_PATH = "bank.csv"   
CSV_SEP = ";"                

print("=== PASO 1: CARGA Y COMPRENSIÓN ===")
df = pd.read_csv(CSV_PATH, sep=CSV_SEP)

print("\nForma del dataset:", df.shape)
print("Columnas:", df.columns.tolist())

print("\nValores faltantes por columna:")
print(df.isna().sum())

if "y" in df.columns:
    print("\nDistribución de 'y':")
    print(df["y"].value_counts())

print("\nVista previa (10 filas):")
print(df.head(10))

# PASO 2 — PREPARACIÓN (One-Hot + Escalado)

print("\n=== PASO 2: PREPARACIÓN ===")

# Variables predictoras (sin 'y')
features = [c for c in df.columns if c != "y"]
X_raw = df[features].copy()

# Identificar tipos
cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_raw.select_dtypes(exclude=["object"]).columns.tolist()

print("Categóricas:", cat_cols)
print("Numéricas:", num_cols)

# Preprocesador (nota: scikit-learn >=1.4 usa 'sparse_output')
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

X = preprocessor.fit_transform(X_raw)

print("\nForma original:", X_raw.shape)
print("Forma transformada:", X.shape)
print("Ejemplo vector (20 primeros valores):", X[0][:20])

# Vector de referencia externa (opcional, para ARI/accuracy por clúster)
y_true = df["y"].map({"no": 0, "yes": 1}).values if "y" in df.columns else None

# PASO 3 — MODELADO Y MÉTRICAS (KMeans, GMM, Birch, DBSCAN muestreado)
print("\n===  PASO 3 — MODELADO Y MÉTRICAS ===")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def metricas_internas(Xmat, labels, sample_size=1000, random_state=42):
    """
    Silhouette sobre muestra (evita O(n^2)), CH y DB sobre todo el conjunto.
    Maneja ruido (-1) de DBSCAN.
    """
    lbls = np.asarray(labels)
    mask = lbls != -1 if np.any(lbls == -1) else np.ones_like(lbls, dtype=bool)
    Xu, Lu = Xmat[mask], lbls[mask]
    if Xu.shape[0] < 2 or np.unique(Lu).size < 2:
        return np.nan, np.nan, np.nan

    ch = calinski_harabasz_score(Xu, Lu)
    db = davies_bouldin_score(Xu, Lu)
    ss = min(sample_size, Xu.shape[0])
    try:
        sil = silhouette_score(Xu, Lu, sample_size=ss, random_state=random_state)
    except Exception:
        sil = np.nan
    return sil, ch, db

def acierto_por_cluster(labels, y):
    """% acierto por clúster vía clase mayoritaria (ignora -1)."""
    if y is None:
        return None, None, None
    lab = np.asarray(labels)
    mask = lab != -1
    lab, yv = lab[mask], np.asarray(y)[mask]
    if lab.size == 0:
        return None, None, None
    dfc = pd.DataFrame({"cluster": lab, "y": yv})
    accs, mapping = {}, {}
    corr, tot = 0, len(dfc)
    for k, g in dfc.groupby("cluster"):
        may = g["y"].mode().iloc[0]
        mapping[int(k)] = int(may)
        acc = (g["y"] == may).mean()
        accs[int(k)] = float(acc)
        corr += (g["y"] == may).sum()
    overall = corr / tot if tot else np.nan
    return accs, overall, mapping

def dbscan_sample_assign(Xmat, eps=1.5, min_samples=10, sample_size=10000, random_state=42):
    """DBSCAN en muestra y asignación del resto por centroides."""
    rng = np.random.default_rng(random_state)
    n = Xmat.shape[0]
    s = min(sample_size, n)
    idx = rng.choice(n, size=s, replace=False)
    Xs = Xmat[idx]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    lbl_s = db.fit_predict(Xs)
    labels = np.full(n, -1, dtype=int)
    labels[idx] = lbl_s

    valid = lbl_s != -1
    if np.unique(lbl_s[valid]).size == 0:
        return labels
    nc = NearestCentroid()
    nc.fit(Xs[valid], lbl_s[valid])
    rest = np.setdiff1d(np.arange(n), idx, assume_unique=True)
    if rest.size:
        labels[rest] = nc.predict(Xmat[rest])
    return labels

# Modelos
modelos = {
    "KMeans": KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE),
    "GMM": GaussianMixture(n_components=3, covariance_type="full", random_state=RANDOM_STATE),
    "Birch": Birch(n_clusters=3, threshold=0.5),
    "DBSCAN_muestra": "dbscan"
}

resultados = []
labels_cache = {}

for nombre, modelo in modelos.items():
    print(f"\n>>> Entrenando {nombre} ...", flush=True)
    if nombre == "DBSCAN_muestra":
        lbls = dbscan_sample_assign(X, eps=1.5, min_samples=10, sample_size=10000, random_state=RANDOM_STATE)
    else:
        lbls = modelo.fit_predict(X)
    labels_cache[nombre] = lbls

    sil, ch, db = metricas_internas(X, lbls, sample_size=1000, random_state=RANDOM_STATE)
    ari = adjusted_rand_score(y_true, lbls) if y_true is not None else np.nan
    accs, overall, mapping = acierto_por_cluster(lbls, y_true)

    resultados.append({
        "modelo": nombre,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "ARI_vs_y": ari,
        "overall_cluster_accuracy": overall
    })

    print(f"--- {nombre} ---")
    print(f"Silhouette: {sil:.4f} | Calinski: {ch:.2f} | Davies: {db:.4f}")
    if not pd.isna(ari):
        print(f"ARI: {ari:.4f}")
    if accs is not None:
        print("Acierto por clúster:", {k: round(v, 4) for k, v in accs.items()})
        print(f"Acierto global: {overall:.4f}")

res_df = pd.DataFrame(resultados).sort_values(by="silhouette", ascending=False)
print("\n=== RESUMEN MODELOS (orden Silhouette) ===")
print(res_df.to_string(index=False))

# PASO 4 — SELECCIÓN + GUARDADO CSV
print("\n=== PASO 4 — SELECCIÓN + GUARDADO ===")
def score_combinado(r):
    s = 0.0
    s += 0 if pd.isna(r["silhouette"]) else r["silhouette"]
    s += 0 if pd.isna(r["calinski_harabasz"]) else r["calinski_harabasz"] / 10000.0
    s += 0 if pd.isna(r["davies_bouldin"]) or r["davies_bouldin"] <= 0 else 1.0 / r["davies_bouldin"]
    s += 0 if pd.isna(r["ARI_vs_y"]) else r["ARI_vs_y"]
    return s

res_df["score_combinado"] = res_df.apply(score_combinado, axis=1)
best_row = res_df.loc[res_df["score_combinado"].idxmax()]
best_model = best_row["modelo"]
print("\nMejor modelo:", best_model)

out = Path("outputs"); out.mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
(res_df.sort_values("score_combinado", ascending=False)
 .to_csv(out / f"metrics_{ts}.csv", index=False))
print("Métricas guardadas en:", out / f"metrics_{ts}.csv")

# PASO 5 — PCA 2D + NUEVOS DATOS EN MARRÓN


print("\n=== PASO 5: PCA + NUEVOS DATOS ===")
NUEVOS_MAX = 200
rng = np.random.default_rng(RANDOM_STATE)

# 5.1 — Panel 2x2 con todos los modelos
pca_all = PCA(n_components=2, random_state=RANDOM_STATE)
Xp = pca_all.fit_transform(X)
fig, axs = plt.subplots(2, 2, figsize=(12, 10)); axs = axs.ravel()
for ax, (n, lab) in zip(axs, labels_cache.items()):
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=lab, s=5)
    ax.set_title(n); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
plt.tight_layout()
plt.savefig(out / f"overview_models_{ts}.png", dpi=160); plt.close()
print("Figura resumen guardada en:", out / f"overview_models_{ts}.png")

# 5.2 — Entrenar en originales y proyectar nuevos datos (marrón)
n_new = min(NUEVOS_MAX, X.shape[0] // 20)
idx_new = rng.choice(X.shape[0], size=n_new, replace=False)
mask_orig = np.ones(X.shape[0], dtype=bool); mask_orig[idx_new] = False
X_orig, X_new = X[mask_orig], X[idx_new]
y_orig = y_true[mask_orig] if y_true is not None else None

from sklearn.neighbors import NearestCentroid

if best_model == "KMeans":
    a = KMeans(n_clusters=len(np.unique(labels_cache[best_model])), n_init=10, random_state=RANDOM_STATE).fit(X_orig)
    lab_orig, lab_new = a.labels_, a.predict(X_new)

elif best_model == "GMM":
    a = GaussianMixture(n_components=len(np.unique(labels_cache[best_model])), random_state=RANDOM_STATE).fit(X_orig)
    lab_orig, lab_new = a.predict(X_orig), a.predict(X_new)

elif best_model == "Birch":
    a = Birch(n_clusters=len(np.unique(labels_cache[best_model])), threshold=0.5).fit(X_orig)
    lab_orig = a.labels_
    try:
        lab_new = a.predict(X_new)
    except Exception:
        nc = NearestCentroid().fit(X_orig, lab_orig)
        lab_new = nc.predict(X_new)

else:  # DBSCAN_muestra
    lab_orig = dbscan_sample_assign(X_orig, eps=1.5, min_samples=10, sample_size=8000, random_state=RANDOM_STATE)
    valid = lab_orig != -1
    if valid.sum() > 0:
        nc = NearestCentroid().fit(X_orig[valid], lab_orig[valid])
        lab_new = nc.predict(X_new)
    else:
        lab_new = np.full(X_new.shape[0], -1, dtype=int)

# 5.3 — Guardar tablas extra si hay y
if y_orig is not None:
    conf = pd.crosstab(pd.Series(lab_orig, name="cluster"), pd.Series(y_orig, name="y"))
    conf.to_csv(out / f"confusion_best_{best_model}_{ts}.csv")
    accs = []
    for k, g in pd.DataFrame({"c": lab_orig, "y": y_orig}).groupby("c"):
        if (g["y"].size):
            accs.append({"cluster": int(k), "accuracy": float((g['y'] == g['y'].mode().iloc[0]).mean())})
    pd.DataFrame(accs).to_csv(out / f"per_cluster_accuracy_{best_model}_{ts}.csv", index=False)

# 5.4 — Gráfico final: nuevos en marrón
pca_final = PCA(n_components=2, random_state=RANDOM_STATE)
X_plot = pca_final.fit_transform(np.vstack([X_orig, X_new]))
Xo, Xn = X_plot[:-len(X_new)], X_plot[-len(X_new):]

plt.figure(figsize=(9, 7))
plt.scatter(Xo[:, 0], Xo[:, 1], c=lab_orig, s=8, alpha=0.75)
plt.scatter(Xn[:, 0], Xn[:, 1], c="brown", s=36, marker="o", label="Nuevos datos")
plt.title(f"Mejor modelo: {best_model} (nuevos datos en marrón)")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.legend()
plt.tight_layout()
plt.savefig(out / f"best_model_with_new_{best_model}_{ts}.png", dpi=160); plt.close()
print("Figura final guardada en:", out / f"best_model_with_new_{best_model}_{ts}.png")

print("\nListo.  './outputs' para CSV/PNG.")
