"""
Pipeline treino -> validação -> teste com rastreamento no MLflow.

Baseado em mlp_torch_avaliacao.py (aula, sousamaf/AI-Lab): mesma rede, perda, otimizador e métricas.
Mudanças: dataset Credit Approval, fit do pré-processador só no treino, mini-batches, seeds fixas,
early stopping pela val_loss e tudo configurável por YAML (+ --set) e registrado no MLflow.

Uso: python -m src.train --config configs/run_a_referencia.yaml [--set treino.lr=0.005]
"""

import argparse
import copy
import os
import tempfile

import joblib
import matplotlib

matplotlib.use("Agg")  # sem janela: figuras vão para arquivo
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.config import achatar_config, carregar_config
from src.data import carregar_dados, descrever_divisao, dividir_dados
from src.features import (
    ajustar_e_transformar,
    criar_preprocessador,
    para_tensores,
    resumo_preprocessamento,
)
from src.model import MLP

NOMES_CLASSES = ["Rejeitado (-)", "Aprovado (+)"]


def fixar_seeds(seed):
    """
    Fixa numpy e torch (o split do sklearn recebe random_state=seed).
    O módulo `random` fica de fora de propósito: o MLflow Tracing o usa para gerar os IDs
    de trace, e com ele fixado todas as runs teriam o mesmo ID.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def escolher_device():
    """mps > cuda > cpu, como na aula."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def avaliar(modelo, X, y, criterio):
    """Perda, acurácia, precision/recall/F1 (macro, como na aula) e ROC-AUC em um conjunto."""
    modelo.eval()
    with torch.no_grad():
        logits = modelo(X)
        perda = criterio(logits, y).item()
        probas = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    return {
        "loss": perda,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, probas),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plotar_curvas(historico, melhor_epoca):
    """Curvas de perda e acurácia (treino x validação), marcando a melhor época."""
    epocas = range(1, len(historico["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epocas, historico["train_loss"], label="Perda de treino")
    ax1.plot(epocas, historico["val_loss"], label="Perda de validação")
    ax1.axvline(melhor_epoca, color="gray", linestyle="--", label=f"Melhor época ({melhor_epoca})")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Perda (cross-entropy)")
    ax1.legend()

    ax2.plot(epocas, historico["train_accuracy"], label="Acurácia de treino")
    ax2.plot(epocas, historico["val_accuracy"], label="Acurácia de validação")
    ax2.axvline(melhor_epoca, color="gray", linestyle="--")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.legend()

    fig.tight_layout()
    return fig


def plotar_matriz_confusao(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NOMES_CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusão (teste)")
    fig.tight_layout()
    return fig


def registrar_modelo(modelo, exemplo_entrada):
    """mlflow.pytorch.log_model mudou de 'artifact_path' (2.x) para 'name' (3.x)."""
    versao_major = int(mlflow.__version__.split(".")[0])
    if versao_major >= 3:
        mlflow.pytorch.log_model(modelo, name="modelo", input_example=exemplo_entrada)
    else:
        mlflow.pytorch.log_model(modelo, artifact_path="modelo", input_example=exemplo_entrada)


def executar(config):
    fixar_seeds(config["seed"])
    device = escolher_device()
    print(f"Usando o dispositivo: {device}")

    cfg_exp, cfg_dados = config["experimento"], config["dados"]
    cfg_modelo, cfg_treino = config["modelo"], config["treino"]

    mlflow.set_tracking_uri(cfg_exp["tracking_uri"])
    mlflow.set_experiment(cfg_exp["nome"])

    with mlflow.start_run(run_name=cfg_exp["run_name"]) as run:
        print(f"Run id: {run.info.run_id}")

        mlflow.set_tags({
            "pergunta": cfg_exp["pergunta"],
            "hipotese": cfg_exp.get("hipotese", ""),
            "dataset": "Credit Approval (UCI id=27)",
            "device": str(device),
            "codigo_base": "AI-Lab/mlp_torch_avaliacao.py",
        })
        mlflow.log_params(achatar_config(config))

        # Um trace por run, com um span por etapa
        with mlflow.start_span(name="pipeline_credit_approval") as span_raiz:
            span_raiz.set_inputs({"config": config})

            # 1) preparar dados
            with mlflow.start_span(name="preparar_dados") as span:
                df, origem = carregar_dados(cfg_dados["caminho_local"], cfg_dados["uci_id"])
                mlflow.set_tag("origem_dados", origem)

                X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(
                    df, cfg_dados["frac_val"], cfg_dados["frac_test"], config["seed"],
                    estratificado=cfg_dados.get("estratificado", True),
                )
                # Idêntico em todas as rodadas com a mesma seed: prova que o split não mudou
                mlflow.log_text(descrever_divisao(X_train, X_val, X_test, y_train, y_val, y_test),
                                "divisao_dados.csv")
                preprocessador = criar_preprocessador()
                Xtr, Xva, Xte = ajustar_e_transformar(preprocessador, X_train, X_val, X_test)

                X_train_t, y_train_t = para_tensores(Xtr, y_train, device)
                X_val_t, y_val_t = para_tensores(Xva, y_val, device)
                X_test_t, y_test_t = para_tensores(Xte, y_test, device)

                # Mini-batches embaralhados (o original era full-batch)
                gerador = torch.Generator().manual_seed(config["seed"])
                loader_treino = DataLoader(
                    TensorDataset(X_train_t, y_train_t),
                    batch_size=cfg_treino["batch_size"], shuffle=True, generator=gerador,
                )

                tamanhos = {
                    "n_total": len(df), "n_train": len(X_train), "n_val": len(X_val),
                    "n_test": len(X_test), "n_features": Xtr.shape[1],
                }
                mlflow.log_params(tamanhos)
                mlflow.log_text(resumo_preprocessamento(preprocessador), "resumo_preprocessamento.txt")
                span.set_inputs({"origem": origem, "split": {k: cfg_dados[k] for k in ("frac_val", "frac_test")}})
                span.set_outputs(tamanhos)
                print(f"Dados: {tamanhos}")

            # 2) treinar
            with mlflow.start_span(name="treinar") as span:
                span.set_inputs(cfg_treino | cfg_modelo)

                modelo = MLP(
                    n_entradas=Xtr.shape[1],
                    camadas_ocultas=cfg_modelo["camadas_ocultas"],
                    n_classes=2,
                    dropout=cfg_modelo["dropout"],
                ).to(device)
                criterio = nn.CrossEntropyLoss()
                otimizador = optim.Adam(
                    modelo.parameters(), lr=cfg_treino["lr"], weight_decay=cfg_treino["weight_decay"]
                )

                historico = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
                melhor_val_loss = float("inf")
                melhor_epoca = 0
                melhores_pesos = copy.deepcopy(modelo.state_dict())
                epocas_sem_melhora = 0

                for epoca in range(1, cfg_treino["epocas"] + 1):
                    modelo.train()
                    soma_perda, acertos, total = 0.0, 0, 0
                    for xb, yb in loader_treino:
                        saidas = modelo(xb)
                        perda = criterio(saidas, yb)

                        otimizador.zero_grad()
                        perda.backward()
                        otimizador.step()

                        soma_perda += perda.item() * len(yb)
                        acertos += (saidas.argmax(dim=1) == yb).sum().item()
                        total += len(yb)
                    train_loss = soma_perda / total
                    train_acc = acertos / total

                    modelo.eval()
                    with torch.no_grad():
                        saidas_val = modelo(X_val_t)
                        val_loss = criterio(saidas_val, y_val_t).item()
                        val_acc = (saidas_val.argmax(dim=1) == y_val_t).float().mean().item()

                    for nome, valor in zip(historico, (train_loss, val_loss, train_acc, val_acc)):
                        historico[nome].append(valor)
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss,
                         "train_accuracy": train_acc, "val_accuracy": val_acc},
                        step=epoca,
                    )

                    # early stopping: guarda os pesos da melhor val_loss
                    if val_loss < melhor_val_loss:
                        melhor_val_loss = val_loss
                        melhor_epoca = epoca
                        melhores_pesos = copy.deepcopy(modelo.state_dict())
                        epocas_sem_melhora = 0
                    else:
                        epocas_sem_melhora += 1

                    if epoca % 10 == 0 or epoca == 1:
                        print(f"Época [{epoca}/{cfg_treino['epocas']}] "
                              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

                    if epocas_sem_melhora >= cfg_treino["paciencia"]:
                        print(f"Early stopping na época {epoca} (melhor época: {melhor_epoca})")
                        break

                epocas_executadas = epoca
                modelo.load_state_dict(melhores_pesos)
                mlflow.log_metrics({"melhor_epoca": melhor_epoca, "epocas_executadas": epocas_executadas})
                mlflow.log_figure(plotar_curvas(historico, melhor_epoca), "curvas_treinamento.png")
                span.set_outputs({"melhor_epoca": melhor_epoca, "epocas_executadas": epocas_executadas,
                                  "melhor_val_loss": melhor_val_loss})

            # 3) validar e testar
            with mlflow.start_span(name="validar_e_testar") as span:
                span.set_inputs({"melhor_epoca": melhor_epoca})

                res_val = avaliar(modelo, X_val_t, y_val_t, criterio)
                mlflow.log_metrics({
                    "val_accuracy_final": res_val["accuracy"], "val_precision": res_val["precision"],
                    "val_recall": res_val["recall"], "val_f1": res_val["f1"], "val_roc_auc": res_val["roc_auc"],
                })

                # O teste é usado uma única vez, aqui, com os pesos da melhor época
                res_test = avaliar(modelo, X_test_t, y_test_t, criterio)
                mlflow.log_metrics({
                    "test_loss": res_test["loss"], "test_accuracy": res_test["accuracy"],
                    "test_precision": res_test["precision"], "test_recall": res_test["recall"],
                    "test_f1": res_test["f1"], "test_roc_auc": res_test["roc_auc"],
                })

                mlflow.log_figure(plotar_matriz_confusao(res_test["y_true"], res_test["y_pred"]),
                                  "matriz_confusao_teste.png")
                relatorio = classification_report(res_test["y_true"], res_test["y_pred"],
                                                  target_names=NOMES_CLASSES, digits=4)
                mlflow.log_text(relatorio, "classification_report_teste.txt")
                mlflow.log_dict(config, "config_usada.yaml")
                with tempfile.TemporaryDirectory() as pasta_tmp:
                    caminho = os.path.join(pasta_tmp, "preprocessador.joblib")
                    joblib.dump(preprocessador, caminho)
                    mlflow.log_artifact(caminho)
                registrar_modelo(modelo, Xva[:5])

                resumo = {
                    "val_f1": round(res_val["f1"], 4), "val_roc_auc": round(res_val["roc_auc"], 4),
                    "test_accuracy": round(res_test["accuracy"], 4), "test_f1": round(res_test["f1"], 4),
                    "test_roc_auc": round(res_test["roc_auc"], 4),
                }
                span.set_outputs(resumo)
                span_raiz.set_outputs(resumo)

        print("\n===== RESULTADO FINAL =====")
        print(f"Melhor época: {melhor_epoca} (executadas: {epocas_executadas})")
        print(f"Validação -> acc={res_val['accuracy']:.4f} f1={res_val['f1']:.4f} roc_auc={res_val['roc_auc']:.4f}")
        print(f"Teste     -> acc={res_test['accuracy']:.4f} precision={res_test['precision']:.4f} "
              f"recall={res_test['recall']:.4f} f1={res_test['f1']:.4f} roc_auc={res_test['roc_auc']:.4f}")
        print(relatorio)
        print(f"Run registrada no MLflow: experimento='{cfg_exp['nome']}', run='{cfg_exp['run_name']}'")


def main():
    parser = argparse.ArgumentParser(description="Treina a MLP de aprovação de crédito com rastreamento no MLflow.")
    parser.add_argument("--config", required=True, help="caminho do YAML do experimento")
    parser.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                        help="sobrescreve um parâmetro do YAML, ex.: --set treino.lr=0.001 (pode repetir)")
    args = parser.parse_args()

    config = carregar_config(args.config, args.set)
    executar(config)


if __name__ == "__main__":
    main()
