# Housing ML — Projeto de Regressão

Projeto simples de Machine Learning: previsão do preço de imóveis (`price`)
a partir de features como área, número de quartos, banheiros, se tem
ar-condicionado, etc. (dataset `Housing.csv`, 545 linhas, sem valores nulos).

## Estrutura

```
housing-ml/
├── data/
│   └── Housing.csv
├── src/
│   ├── eda.py            # análise exploratória (colunas, nulos, correlações)
│   ├── preprocess.py      # encoding + split + padronização (compartilhado)
│   ├── train_tree.py      # Random Forest Regressor (scikit-learn)
│   └── train_mlp.py       # MLP (PyTorch)
├── outputs/                # gerado ao rodar os scripts (métricas, modelos, gráficos)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Metodologia (atualizada com o feedback do professor)

**Split dos dados (3 partes):**
- **Treino (70%)**: usado para calcular o gradiente e ajustar os pesos do modelo
- **Validação (15%)**: usado só para o *early stopping* (decidir quando parar), nunca ajusta pesos
- **Teste cego (15%)**: nunca visto durante treino nem validação; avaliado uma única vez, no final

Random Forest não usa gradiente/early stopping, então treina com treino+validação
juntos e avalia no mesmo teste cego (para a comparação com a MLP ser justa).

**Treino da MLP:**
- Pré-treino: inicialização dos pesos (default do PyTorch) + padronização das
  features (média 0, desvio 1) calculada só a partir do treino
- Cada batch: MSE loss -> `loss.backward()` (backpropagation/autograd calcula o
  gradiente) -> `optimizer.step()` (Adam atualiza os pesos)
- Ao fim de cada época: loss de validação é calculada (sem atualizar pesos)
- **Early stopping com patience=20**: se a validação não melhora por 20 épocas
  seguidas, o treino para e os pesos da melhor época são restaurados

**Explicabilidade (XAI):** SHAP (`shap.TreeExplainer` para a árvore,
`shap.KernelExplainer` para a MLP) para ver quais features mais influenciam
cada previsão individual, além da importância nativa (MDI) do Random Forest.

## Rodando sem Docker (ambiente local com Python 3.11)

```bash
pip install -r requirements.txt

cd src
python eda.py           # gera outputs/correlation_heatmap.png, price_distribution.png
python train_tree.py    # treina a árvore, salva outputs/tree_model.joblib
python train_mlp.py     # treina a MLP, salva outputs/mlp_model.pt
```

**Novos arquivos gerados em `outputs/`:**
- `tree_scatter_fit.png` / `mlp_scatter_fit.png`: preço real vs. previsto + reta de ajuste
- `mlp_training_curve.png`: loss de treino e validação por época (mostra onde o early stopping parou)
- `tree_shap_summary.png` / `mlp_shap_summary.png`: importância das features via SHAP
- `tree_metrics.json` / `mlp_metrics.json`: agora incluem `split_sizes`, `history` (MLP) e `shap_mean_abs`

> **Nota sobre tempo**: o cálculo do SHAP para a MLP usa `KernelExplainer`,
> que é mais lento (precisa rodar o modelo várias vezes por amostra). Pode
> levar alguns minutos, mesmo rodando só numa amostra de 30 exemplos do teste.

## Rodando com Docker

1. Build da imagem:
```bash
docker build -t housing-ml .
```

2. Rodar o treino da árvore (padrão) e salvar os resultados na sua máquina:
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs housing-ml
```

3. Rodar a MLP em vez da árvore:
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs housing-ml python train_mlp.py
```

4. Rodar a EDA:
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs housing-ml python eda.py
```

> O `-v $(pwd)/outputs:/app/outputs` monta a pasta `outputs/` local dentro do
> container, então os arquivos gerados (métricas, modelos, gráficos) aparecem
> na sua máquina depois do container terminar.

## Resultados obtidos (rodados localmente, fora do Docker)

> ⚠️ Os splits mudaram (agora treino/validação/teste cego em vez de treino/teste
> simples), então os números abaixo **não são diretamente comparáveis** aos que
> você tinha rodado antes. Rode de novo pra pegar os valores atualizados.

| Modelo         | RMSE          | MAE           | R²    |
|----------------|---------------|---------------|-------|
| Random Forest  | ~1.31M        | ~0.95M        | ~0.59 |
| MLP (PyTorch)  | *a testar*    | *a testar*    | *a testar* |

## Próximos passos

- [ ] Rodar `train_mlp.py` de novo (com early stopping) e conferir as métricas e a curva de treino
- [ ] Se quiser a análise SHAP, instalar a lib: `pip install shap --break-system-packages`
      (ou reconstruir a imagem Docker, que já inclui `shap` no `requirements.txt`)
- [ ] Testar o build/run do Docker de novo -- **precisa rebuildar** (`docker build -t housing-ml .`)
      porque o `requirements.txt` mudou (adicionamos `shap`)
- [ ] Depois disso: atualizar os slides do Overleaf com a nova metodologia e os resultados
