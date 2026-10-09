#!/usr/bin/env bash
# Instala o ambiente do projeto no Linux/macOS (cria .venv e instala as dependências).
# Uso:  bash setup.sh
# Depois:  source .venv/bin/activate  e  bash run_all.sh
set -e

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 não encontrado. No Ubuntu: sudo apt install python3 python3-venv python3-pip"
  exit 1
fi

echo "== Python: $(python3 --version)"
python3 -m venv .venv || { echo "Falhou ao criar o venv. No Ubuntu: sudo apt install python3-venv"; exit 1; }
source .venv/bin/activate

python -m pip install --upgrade pip --quiet
echo "== Instalando torch (versão CPU)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
echo "== Instalando o restante (mlflow, scikit-learn, pandas...)..."
pip install -r requirements.txt --quiet

python -c "import torch, mlflow, sklearn; print('== OK: torch', torch.__version__, '| mlflow', mlflow.__version__, '| sklearn', sklearn.__version__)"
echo
echo "Pronto. Agora rode:"
echo "  source .venv/bin/activate"
echo "  bash run_all.sh"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db     # depois abra http://127.0.0.1:5000"
