# Executa as cinco rodadas em sequencia (Windows / PowerShell).
# Uso:  .\run_all.ps1          (com o ambiente virtual ja ativado)
$ErrorActionPreference = "Continue"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

Write-Host "=== Rodada A: referencia (lr 0.01, L2 0) ==="
python -m src.train --config configs/run_a_referencia.yaml

Write-Host "=== Rodada B: lr menor (lr 0.001) ==="
python -m src.train --config configs/run_b_lr_menor.yaml

Write-Host "=== Rodada C: com L2 (weight_decay 1e-5) ==="
python -m src.train --config configs/run_c_com_l2.yaml

Write-Host "=== Rodada D: MLP grande [128, 64] (investigacao extra) ==="
python -m src.train --config configs/run_d_mlp_grande.yaml

Write-Host "=== Rodada E: MLP grande + dropout 0.3 (nova run apos o diagnostico de D) ==="
python -m src.train --config configs/run_e_mlp_grande_dropout.yaml

Write-Host "=== Tabela comparativa ==="
python -m src.comparar
