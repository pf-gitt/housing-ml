"""
Treino do modelo baseado em árvore (Random Forest Regressor).

Sobre o split: Random Forest não é treinado por gradiente e não usa early
stopping -- então não precisamos "gastar" dados com validação durante o
ajuste do modelo. Por isso, treinamos com (treino + validação) juntos e
avaliamos SÓ UMA VEZ no teste cego, mantendo a mesma partição de teste cego
usada pela MLP (mesmo random_state), para a comparação entre os dois
modelos ser justa.
"""
import json
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import load_and_prepare

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def plot_scatter_with_fit(y_true, y_pred, title, save_path):
    """Scatter preço real (Y) vs. preço previsto (X) + reta de ajuste linear
    (regressão) + linha de referência y=x (previsão perfeita)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, y_true, alpha=0.6, edgecolor="k", linewidth=0.3, color="#7A1F2D")

    # Reta de ajuste linear: real em função do previsto (polyfit grau 1)
    coeffs = np.polyfit(y_pred, y_true, deg=1)
    fit_line = np.poly1d(coeffs)
    x_range = np.linspace(y_pred.min(), y_pred.max(), 100)
    ax.plot(x_range, fit_line(x_range), color="#7A1F2D", linewidth=2,
             label=f"Ajuste linear (y={coeffs[0]:.2f}x+{coeffs[1]:.0f})")

    # Linha de referência: previsão perfeita (y = x)
    full_range = np.linspace(min(y_pred.min(), y_true.min()),
                              max(y_pred.max(), y_true.max()), 100)
    ax.plot(full_range, full_range, color="gray", linestyle="--", linewidth=1.5,
             label="Previsão perfeita (y=x)")

    ax.set_xlabel("Preço previsto")
    ax.set_ylabel("Preço real")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    data = load_and_prepare(scale_target=False)
    print(f"Split -> treino: {data['split_sizes']['train']}, "
          f"validação: {data['split_sizes']['val']}, "
          f"teste cego: {data['split_sizes']['test']}")

    # Random Forest: treina com treino + validação (ver docstring do módulo)
    X_fit = np.vstack([data["X_train"], data["X_val"]])
    y_fit = np.concatenate([data["y_train"], data["y_val"]])
    X_test, y_test = data["X_test"], data["y_test"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)

    # ---- Avaliação no TESTE CEGO (usado uma única vez) ----
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"model": "RandomForestRegressor", "rmse": rmse, "mae": mae, "r2": r2}
    print(json.dumps(metrics, indent=2))

    # Scatter plot com fit linear
    plot_scatter_with_fit(
        y_test, preds,
        title="Random Forest: Preço Real vs. Previsto (teste cego)",
        save_path=OUT_DIR / "tree_scatter_fit.png",
    )

    # Importância nativa das features (MDI - Mean Decrease in Impurity)
    importances = dict(zip(data["feature_names"], model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # ---- SHAP (explicabilidade / XAI) ----
    shap_summary = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary = dict(zip(data["feature_names"], mean_abs_shap.tolist()))
        shap_summary = dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values, X_test, feature_names=data["feature_names"],
            plot_type="bar", show=False
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "tree_shap_summary.png", dpi=150)
        plt.close()
        print(f"SHAP summary salvo em {OUT_DIR / 'tree_shap_summary.png'}")
    except ImportError:
        print("\n[AVISO] biblioteca 'shap' não instalada -- pulei a análise de "
              "explicabilidade. Rode: pip install shap --break-system-packages")

    with open(OUT_DIR / "tree_metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "split_sizes": data["split_sizes"],
            "feature_importances": importances,
            "shap_mean_abs": shap_summary,
        }, f, indent=2)

    joblib.dump(model, OUT_DIR / "tree_model.joblib")
    print(f"\nModelo salvo em {OUT_DIR / 'tree_model.joblib'}")
    print(f"Métricas salvas em {OUT_DIR / 'tree_metrics.json'}")


if __name__ == "__main__":
    main()
