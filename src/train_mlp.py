"""
Treino do modelo MLP (PyTorch) para regressão.

Fluxo de treino (resumo p/ os slides):
- Pré-treino: os pesos das camadas Linear são inicializados automaticamente
  pelo PyTorch (Kaiming Uniform), e as features de entrada são padronizadas
  (média 0, desvio 1) ANTES do treino começar -- isso ajuda o gradiente a
  convergir mais rápido e de forma mais estável.
- Treino: a cada batch, calculamos a MSE (erro quadrático médio) entre a
  previsão e o valor real, e usamos backpropagation (autograd do PyTorch)
  para calcular o gradiente da loss em relação a cada peso da rede. O
  otimizador Adam usa esse gradiente para atualizar os pesos.
- Validação: ao final de cada época, avaliamos a loss no conjunto de
  validação (sem atualizar pesos). Essa loss NÃO influencia o gradiente,
  só é usada para decidir quando parar (early stopping).
- Early stopping (patience): se a loss de validação não melhorar por
  `patience` épocas seguidas, o treino para e voltamos para os pesos da
  melhor época (evita overfitting).
- Teste cego: só depois de todo o treino/validação terminarem, avaliamos
  UMA ÚNICA VEZ no conjunto de teste cego -- que nunca influenciou nem os
  pesos nem a decisão de parada.
"""
import copy
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import load_and_prepare

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class MLPRegressor(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def plot_training_curve(history, save_path):
    plt.figure(figsize=(7, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Treino", color="#333333")
    plt.plot(epochs, history["val_loss"], label="Validação", color="#7A1F2D")
    if history.get("best_epoch"):
        plt.axvline(history["best_epoch"], color="gray", linestyle="--",
                     label=f"Melhor época ({history['best_epoch']})")
    plt.xlabel("Época")
    plt.ylabel("Loss (MSE, escala padronizada)")
    plt.title("Curva de treinamento por época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scatter_with_fit(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3, color="#7A1F2D")

    coeffs = np.polyfit(y_true, y_pred, deg=1)
    fit_line = np.poly1d(coeffs)
    x_range = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_range, fit_line(x_range), color="#7A1F2D", linewidth=2,
             label=f"Ajuste linear (y={coeffs[0]:.2f}x+{coeffs[1]:.0f})")
    ax.plot(x_range, x_range, color="gray", linestyle="--", linewidth=1.5,
             label="Previsão perfeita (y=x)")

    ax.set_xlabel("Preço real")
    ax.set_ylabel("Preço previsto")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(max_epochs=500, patience=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_and_prepare(scale_target=True)
    print(f"Split -> treino: {data['split_sizes']['train']}, "
          f"validação: {data['split_sizes']['val']}, "
          f"teste cego: {data['split_sizes']['test']}")

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    scaler_y = data["scaler_y"]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # "Pré-treino": inicialização default do PyTorch (Kaiming Uniform) já
    # acontece aqui, na construção do modelo.
    model = MLPRegressor(n_features=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "best_epoch": None}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # ---- Treino: cálculo do gradiente via backpropagation ----
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()       # calcula dLoss/dPeso para cada peso (autograd)
            optimizer.step()      # Adam atualiza os pesos usando o gradiente
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        # ---- Validação: só avalia, não atualiza pesos ----
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device))
            val_loss = float(criterion(val_preds, y_val_t.to(device)).item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ---- Early stopping com patience ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{max_epochs} - train_loss: {train_loss:.4f} "
                  f"- val_loss: {val_loss:.4f} - sem melhora há {epochs_without_improvement} épocas")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping na época {epoch} "
                  f"(sem melhora na validação por {patience} épocas seguidas)")
            break

    # Restaura os pesos da melhor época (menor val_loss), não os da última
    model.load_state_dict(best_state)
    print(f"Restaurando pesos da melhor época: {history['best_epoch']}")

    # ---- Avaliação no TESTE CEGO (usado uma única vez, nunca antes) ----
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t.to(device)).cpu().numpy()

    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_orig, preds)))
    mae = float(mean_absolute_error(y_test_orig, preds))
    r2 = float(r2_score(y_test_orig, preds))

    metrics = {"model": "MLP (PyTorch)", "rmse": rmse, "mae": mae, "r2": r2,
               "best_epoch": history["best_epoch"], "stopped_at_epoch": epoch,
               "patience": patience}
    print(json.dumps(metrics, indent=2))

    # Curva de treino por época
    plot_training_curve(history, OUT_DIR / "mlp_training_curve.png")

    # Scatter plot com fit linear
    plot_scatter_with_fit(
        y_test_orig, preds,
        title="MLP: Preço Real vs. Previsto (teste cego)",
        save_path=OUT_DIR / "mlp_scatter_fit.png",
    )

    # ---- SHAP (explicabilidade / XAI) ----
    shap_summary = None
    try:
        import shap
        model.eval()

        def predict_fn(x_numpy):
            with torch.no_grad():
                x_t = torch.tensor(x_numpy, dtype=torch.float32).to(device)
                return model(x_t).cpu().numpy()

        # Background = amostra do treino; explica uma amostra do teste cego
        background = X_train[np.random.choice(X_train.shape[0], size=50, replace=False)]
        explainer = shap.KernelExplainer(predict_fn, background)
        sample_test = X_test[:30]  # amostra p/ manter o custo computacional razoável
        shap_values = explainer.shap_values(sample_test, nsamples=100)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_summary = dict(zip(data["feature_names"], mean_abs_shap.tolist()))
        shap_summary = dict(sorted(shap_summary.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values, sample_test, feature_names=data["feature_names"],
            plot_type="bar", show=False
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mlp_shap_summary.png", dpi=150)
        plt.close()
        print(f"SHAP summary salvo em {OUT_DIR / 'mlp_shap_summary.png'}")
    except ImportError:
        print("\n[AVISO] biblioteca 'shap' não instalada -- pulei a análise de "
              "explicabilidade. Rode: pip install shap --break-system-packages")

    with open(OUT_DIR / "mlp_metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "split_sizes": data["split_sizes"],
            "history": history,
            "shap_mean_abs": shap_summary,
        }, f, indent=2)

    torch.save(model.state_dict(), OUT_DIR / "mlp_model.pt")
    print(f"\nModelo salvo em {OUT_DIR / 'mlp_model.pt'}")
    print(f"Métricas salvas em {OUT_DIR / 'mlp_metrics.json'}")


if __name__ == "__main__":
    main()
