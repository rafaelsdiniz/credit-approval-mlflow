# Executa todos os experimentos em sequência (Windows / PowerShell).
# Uso:  .\run_all.ps1          (com o ambiente virtual já ativado)
$ErrorActionPreference = "Continue"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

Write-Host "=== exp01: MLP pequena [16] ==="
python -m src.train --config configs/exp01_mlp_pequena.yaml

Write-Host "=== exp02: MLP grande [128, 64] ==="
python -m src.train --config configs/exp02_mlp_grande.yaml

Write-Host "=== exp02 + dropout 0.3 (regularização) ==="
python -m src.train --config configs/exp02_mlp_grande.yaml --set modelo.dropout=0.3 --set experimento.run_name=mlp_128_64_dropout03

Write-Host "=== exp02 + weight_decay 1e-3 (regularização L2) ==="
python -m src.train --config configs/exp02_mlp_grande.yaml --set treino.weight_decay=1e-3 --set experimento.run_name=mlp_128_64_wd1e-3

Write-Host "=== Tabela comparativa ==="
python -m src.comparar
