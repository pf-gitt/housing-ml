# Imagem simples para rodar EDA e treino dos modelos de regressão
FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema (necessárias para matplotlib/seaborn renderizarem sem display)
ENV MPLBACKEND=Agg

# Instala dependências Python primeiro (melhora cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código e os dados
COPY src/ ./src/
COPY data/ ./data/

WORKDIR /app/src

# Por padrão roda o treino da árvore; pode ser sobrescrito no `docker run`
# Ex: docker run housing-ml python train_mlp.py
CMD ["python", "train_tree.py"]
