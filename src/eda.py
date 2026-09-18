"""
Análise Exploratória de Dados (EDA) - Housing Dataset
Etapa 1 do projeto: entender colunas, valores nulos e correlações.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Housing.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("1. FORMATO E COLUNAS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("2. VALORES NULOS")
    print("=" * 60)
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "Nenhum valor nulo encontrado.")

    print("\n" + "=" * 60)
    print("3. ESTATÍSTICAS DESCRITIVAS (numéricas)")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("4. VALORES ÚNICOS (colunas categóricas)")
    print("=" * 60)
    cat_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in cat_cols:
        print(f"{col}: {df[col].unique().tolist()}")

    # Codifica yes/no como 1/0 e furnishingstatus como ordinal só p/ ver correlação
    df_corr = df.copy()
    yes_no_cols = [c for c in cat_cols if set(df[c].unique()) <= {"yes", "no"}]
    for col in yes_no_cols:
        df_corr[col] = df_corr[col].map({"yes": 1, "no": 0})

    if "furnishingstatus" in df_corr.columns:
        order = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
        df_corr["furnishingstatus"] = df_corr["furnishingstatus"].map(order)

    print("\n" + "=" * 60)
    print("5. CORRELAÇÃO COM O TARGET (price)")
    print("=" * 60)
    corr_with_price = df_corr.corr(numeric_only=True)["price"].sort_values(ascending=False)
    print(corr_with_price)

    # Heatmap de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de Correlação - Housing Dataset")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"\nHeatmap salvo em: {OUT_DIR / 'correlation_heatmap.png'}")

    # Distribuição do target
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], kde=True)
    plt.title("Distribuição do Preço (target)")
    plt.xlabel("price")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "price_distribution.png", dpi=150)
    plt.close()
    print(f"Distribuição salva em: {OUT_DIR / 'price_distribution.png'}")

    # Salva resumo em CSV pra usar depois nos slides
    corr_with_price.to_csv(OUT_DIR / "correlation_with_price.csv")


if __name__ == "__main__":
    main()
