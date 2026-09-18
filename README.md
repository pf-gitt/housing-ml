# Housing ML — Previsão de Preços de Imóveis

Projeto de Machine Learning (regressão): previsão do preço de imóveis
(`price`) a partir de features como área, número de quartos, banheiros,
ar-condicionado, etc. Dataset `Housing.csv` (545 imóveis, 13 colunas, sem
valores nulos). Dois modelos são treinados e comparados: **Random Forest**
(scikit-learn) e uma **MLP** (PyTorch).

## Estrutura

```
housing-ml/
├── data/
│   └── Housing.csv
├── src/
│   ├── eda.py            # análise exploratória (colunas, nulos, correlações)
│   ├── preprocess.py     # encoding + split treino/val/teste + padronização
│   ├── train_tree.py     # Random Forest Regressor (scikit-learn)
│   └── train_mlp.py      # MLP (PyTorch), com early stopping
├── outputs/               # métricas, modelos e gráficos gerados pelos scripts
├── Dockerfile
├── requirements.txt
└── README.md
```

## Metodologia

**Split dos dados em 3 partes** (evita vazamento de informação):
- **Treino (70%, 381 amostras)**: usado para calcular o gradiente e ajustar os pesos do modelo
- **Validação (15%, 82 amostras)**: usado só para o *early stopping* (decidir quando parar), nunca ajusta pesos
- **Teste cego (15%, 82 amostras)**: nunca visto durante treino nem validação; avaliado uma única vez, no final

Random Forest não usa gradiente nem early stopping, então treina com
treino+validação juntos e avalia no mesmo teste cego (mantendo a comparação
com a MLP justa).

**Treino da MLP:**
- Pré-treino: inicialização dos pesos (default do PyTorch, Kaiming) + padronização
  das features (média 0, desvio 1) calculada só a partir do treino
- Cada batch: MSE loss → `loss.backward()` (backpropagation/autograd calcula o
  gradiente) → `optimizer.step()` (Adam atualiza os pesos)
- Ao fim de cada época: loss de validação é calculada (sem atualizar pesos)
- **Early stopping com patience=20**: se a validação não melhora por 20 épocas
  seguidas, o treino para e os pesos da melhor época são restaurados

**Explicabilidade (XAI):** SHAP (`shap.TreeExplainer` para a árvore,
`shap.KernelExplainer` para a MLP), para medir o impacto de cada feature nas
previsões, complementando a importância nativa (MDI) do Random Forest.

## Rodando sem Docker

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd src
python eda.py           # gera outputs/correlation_heatmap.png, price_distribution.png
python train_tree.py    # treina a árvore, salva outputs/tree_model.joblib
python train_mlp.py     # treina a MLP, salva outputs/mlp_model.pt
```

**Arquivos gerados em `outputs/`:**
- `correlation_heatmap.png`, `price_distribution.png`: análise exploratória
- `tree_scatter_fit.png` / `mlp_scatter_fit.png`: preço real vs. previsto + reta de ajuste
- `mlp_training_curve.png`: loss de treino e validação por época
- `tree_shap_summary.png` / `mlp_shap_summary.png`: importância das features via SHAP
- `tree_metrics.json` / `mlp_metrics.json`: métricas, `split_sizes`, histórico de treino (MLP) e valores SHAP

> O cálculo do SHAP para a MLP usa `KernelExplainer`, que é mais lento
> (roda o modelo várias vezes por amostra) — pode levar alguns minutos.

## Rodando com Docker

Build da imagem:
```bash
docker build -t housing-ml .
```

Rodar o treino da árvore (padrão) e salvar os resultados na sua máquina:
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs housing-ml
```
*(Windows/cmd: troque `$(pwd)` por `%cd%`)*

Rodar a MLP:
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs housing-ml python train_mlp.py
```

Rodar a EDA:
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs housing-ml python eda.py
```

> O `-v $(pwd)/outputs:/app/outputs` monta a pasta `outputs/` local dentro do
> container, então os arquivos gerados aparecem na sua máquina depois do
> container terminar.

## Resultados (teste cego, 82 amostras nunca vistas no treino/validação)

| Modelo         | RMSE   | MAE    | R²    |
|----------------|--------|--------|-------|
| Random Forest  | 1.31M  | 0.95M  | 0.592 |
| MLP (PyTorch)  | 1.19M  | 0.90M  | 0.662 |

A MLP superou o Random Forest nas três métricas. O early stopping da MLP
disparou cedo (melhor época: 10; parada na época 30), indicando que o
dataset (545 amostras) é pequeno para uma rede neural — a loss de treino
continua caindo após a época 10, mas a de validação estagna, sinal de
overfitting.

Em ambos os modelos, `area` e `bathrooms` são as features mais importantes,
tanto pela importância nativa do Random Forest quanto pelos valores SHAP.
