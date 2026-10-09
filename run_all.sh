#!/usr/bin/env bash
# Executa as cinco rodadas em sequência (Linux / macOS / Git Bash).
# Uso:  bash run_all.sh       (com o ambiente virtual já ativado)
export MLFLOW_DISABLE_AGENT_HINT=1

echo "=== Rodada A: referencia (lr 0.01, L2 0) ==="
python -m src.train --config configs/run_a_referencia.yaml

echo "=== Rodada B: lr menor (lr 0.001) ==="
python -m src.train --config configs/run_b_lr_menor.yaml

echo "=== Rodada C: com L2 (weight_decay 1e-5) ==="
python -m src.train --config configs/run_c_com_l2.yaml

echo "=== Rodada D: MLP grande [128, 64] (investigacao extra) ==="
python -m src.train --config configs/run_d_mlp_grande.yaml

echo "=== Rodada E: MLP grande + dropout 0.3 (nova run apos o diagnostico de D) ==="
python -m src.train --config configs/run_e_mlp_grande_dropout.yaml

echo "=== Tabela comparativa ==="
python -m src.comparar
