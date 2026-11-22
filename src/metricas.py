"""Métricas y gráficas de evaluación (precision, recall, F1, matriz de confusión, ROC/AUC, PR).

Convención: y_true = 1 si la imagen es "danado" (clase positiva), 0 si es "sin_dano".
score = probabilidad que el modelo asigna a "danado".
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from . import config

CLASS_NAMES = [config.TAG_LABELS[config.NEGATIVE_TAG], config.TAG_LABELS[config.POSITIVE_TAG]]
PALETTE = ["#2a6fdb", "#e8833a", "#3aa76d", "#8e5cc9"]


def compute_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else float("nan"),
        "average_precision": float(average_precision_score(y_true, scores)) if both_classes else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "classification_report": report,
    }


def bootstrap_auc_ci(y_true, scores, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Intervalo de confianza del AUC por bootstrap (remuestreo de imágenes de prueba)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def threshold_analysis(y_true, scores, target_recall: float = 0.95) -> dict:
    """Umbrales de decisión candidatos: Youden (ROC), F1 máximo y recall objetivo."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(y_true, scores)
    finite = np.isfinite(thr)
    j = tpr - fpr
    j[~finite] = -1
    youden_thr = float(thr[int(np.argmax(j))])

    prec, rec, thr_pr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    f1_thr = float(thr_pr[int(np.argmax(f1))])

    ok = np.where(rec[:-1] >= target_recall)[0]
    recall_thr = float(thr_pr[ok[-1]]) if len(ok) else float(thr_pr[0])
    return {"youden": youden_thr, "max_f1": f1_thr, f"recall_{int(target_recall*100)}": recall_thr}


# ----------------------------------------------------------------------------- gráficas

def plot_confusion(y_true, scores, threshold=0.5, title="Matriz de confusión"):
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.6 else "#1b1b1b"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                    color=color, fontsize=11)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"{title}\n(umbral = {threshold:.2f})", fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc(curves: dict, title="Curva ROC (clase positiva: con daño)"):
    """curves = {nombre: (y_true, scores)}"""
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    for k, (name, (y, s)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[k % len(PALETTE)],
                label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="Azar (AUC = 0.5)")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_pr(curves: dict, title="Curva Precision–Recall"):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    for k, (name, (y, s)) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, lw=2, color=PALETTE[k % len(PALETTE)],
                label=f"{name} (AP = {average_precision_score(y, s):.3f})")
    base = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(base, ls="--", color="#999999", lw=1, label=f"Línea base ({base:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(y_true, scores, marks: dict | None = None,
                         title="Precision, recall y F1 según el umbral"):
    ts = np.linspace(0.01, 0.99, 99)
    p, r, f = [], [], []
    for t in ts:
        y_pred = (np.asarray(scores) >= t).astype(int)
        p.append(precision_score(y_true, y_pred, zero_division=0))
        r.append(recall_score(y_true, y_pred, zero_division=0))
        f.append(f1_score(y_true, y_pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    ax.plot(ts, p, lw=2, color=PALETTE[0], label="Precision")
    ax.plot(ts, r, lw=2, color=PALETTE[1], label="Recall")
    ax.plot(ts, f, lw=2, color=PALETTE[2], label="F1")
    # etiquetas escalonadas en altura para que no se solapen si los umbrales están cerca
    for i, (name, t) in enumerate(sorted((marks or {}).items(), key=lambda kv: kv[1])):
        ax.axvline(t, ls=":", color="#555555", lw=1)
        ax.text(t + 0.01, 0.05 + 0.13 * i, f"{name} = {t:.2f}", fontsize=7, color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_xlabel("Umbral sobre P(con daño)")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_score_hist(y_true, scores, threshold=0.5, title="Distribución de P(con daño) por clase real"):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=150)
    bins = np.linspace(0, 1, 26)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.7, color=PALETTE[0], label=CLASS_NAMES[0])
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.7, color=PALETTE[1], label=CLASS_NAMES[1])
    ax.axvline(threshold, ls="--", color="#333333", lw=1, label=f"Umbral {threshold:.2f}")
    ax.set_xlabel("P(con daño)")
    ax.set_ylabel("Número de imágenes")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
