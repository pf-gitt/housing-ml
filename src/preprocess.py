"""
Pré-processamento compartilhado entre os modelos.

Split em 3 partes:
- treino (train): usado para calcular o gradiente / ajustar os parâmetros do modelo
- validação (val): usado para early stopping (patience) -- NÃO é usado para ajustar pesos
- teste cego (blind test): NUNCA visto durante treino nem validação.
  Só é usado 1x, no final, para reportar a métrica final do modelo.

Proporção default: 70% treino / 15% validação / 15% teste cego.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Housing.csv"

YES_NO_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]
TARGET = "price"


def load_and_prepare(
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    scale_target=False,
):
    """
    Carrega o CSV, codifica categóricas, padroniza numéricas e faz o split
    em 3 partes (treino / validação / teste cego).

    A separação é feita em duas etapas:
      1) separa o teste cego (test_size) do resto
      2) do que sobrou, separa a validação (val_size, proporcional ao total original)
    """
    df = pd.read_csv(DATA_PATH)

    # yes/no -> 1/0
    for col in YES_NO_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # furnishingstatus -> one-hot
    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    # Etapa 1: separa o teste cego
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Etapa 2: separa treino e validação do que sobrou
    # val_size é relativo ao dataset ORIGINAL, então recalculamos a proporção
    # relativa ao X_temp (que já não tem o teste cego)
    val_ratio_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, random_state=random_state
    )

    # Padroniza features numéricas com base APENAS no treino
    # (evita vazamento de informação de val/teste para o treino)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = None
    y_train_out, y_val_out, y_test_out = y_train.values, y_val.values, y_test.values
    if scale_target:
        scaler_y = StandardScaler()
        y_train_out = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_out = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_out = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_out,
        "y_val": y_val_out,
        "y_test": y_test_out,
        "feature_names": X.columns.tolist(),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "split_sizes": {
            "train": len(X_train), "val": len(X_val), "test": len(X_test),
        },
    }
