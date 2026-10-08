#!/usr/bin/env bash
# Executa todos os experimentos em sequência (Linux / macOS / Git Bash).
# Uso:  bash run_all.sh       (com o ambiente virtual já ativado)
export MLFLOW_DISABLE_AGENT_HINT=1

echo "=== exp01: MLP pequena [16] ==="
python -m src.train --config configs/exp01_mlp_pequena.yaml

echo "=== exp02: MLP grande [128, 64] ==="
python -m src.train --config configs/exp02_mlp_grande.yaml

echo "=== exp02 + dropout 0.3 (regularização) ==="
python -m src.train --config configs/exp02_mlp_grande.yaml --set modelo.dropout=0.3 --set experimento.run_name=mlp_128_64_dropout03

echo "=== exp02 + weight_decay 1e-3 (regularização L2) ==="
python -m src.train --config configs/exp02_mlp_grande.yaml --set treino.weight_decay=1e-3 --set experimento.run_name=mlp_128_64_wd1e-3

echo "=== Tabela comparativa ==="
python -m src.comparar
