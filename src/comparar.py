"""
Imprime uma tabela (Markdown) com as runs de todos os experimentos registrados no MLflow.

Por quê: facilita comparar as runs lado a lado no terminal e copiar os números
reais para o README sem digitar nada à mão.

Uso:
  python -m src.comparar                      # usa sqlite:///mlflow.db
  python -m src.comparar --tracking-uri ...   # outro servidor/arquivo
"""

import argparse

import mlflow
import pandas as pd

COLUNAS = {
    "experimento": "Experimento",
    "tags.mlflow.runName": "Run",
    "params.modelo.camadas_ocultas": "Camadas",
    "params.modelo.dropout": "Dropout",
    "params.treino.weight_decay": "L2",
    "params.treino.lr": "lr",
    "metrics.melhor_epoca": "Melhor época",
    "metrics.val_f1": "val_f1",
    "metrics.val_roc_auc": "val_roc_auc",
    "metrics.test_accuracy": "test_accuracy",
    "metrics.test_f1": "test_f1",
    "metrics.test_roc_auc": "test_roc_auc",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experimentos = mlflow.search_experiments()
    nomes = [e.name for e in experimentos if e.name != "Default"]
    if not nomes:
        print("Nenhum experimento encontrado. Rode primeiro: python -m src.train --config configs/run_a_referencia.yaml")
        return

    runs = mlflow.search_runs(experiment_names=nomes, filter_string="attributes.status = 'FINISHED'")
    if runs.empty:
        print("Nenhuma run finalizada encontrada.")
        return

    # search_runs devolve o id do experimento; trocamos pelo nome para ficar legível
    id_para_nome = {e.experiment_id: e.name for e in experimentos}
    runs["experimento"] = runs["experiment_id"].map(id_para_nome)

    presentes = [c for c in COLUNAS if c in runs.columns]
    tabela = runs[presentes].rename(columns=COLUNAS).sort_values(["Experimento", "Run"])
    for col in tabela.columns:
        if tabela[col].dtype.kind == "f":
            tabela[col] = tabela[col].round(4)
    if "Melhor época" in tabela.columns:
        tabela["Melhor época"] = tabela["Melhor época"].astype(int)

    pd.set_option("display.width", 200)
    print(tabela.to_markdown(index=False))


if __name__ == "__main__":
    main()
