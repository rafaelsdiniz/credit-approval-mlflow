"""
Pré-processamento das features (imputação, padronização e one-hot).
Correção em relação à aula: o fit acontece SÓ no treino, para não vazar validação/teste.
"""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import COLUNAS_CATEGORICAS, COLUNAS_NUMERICAS


def criar_preprocessador():
    """Numéricos: mediana + StandardScaler. Categóricos: moda + OneHotEncoder (ignora categoria nova no teste)."""
    pipeline_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("padronizar", StandardScaler()),
    ])
    pipeline_categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", pipeline_numerico, COLUNAS_NUMERICAS),
        ("cat", pipeline_categorico, COLUNAS_CATEGORICAS),
    ])


def ajustar_e_transformar(preprocessador, X_train, X_val, X_test):
    """fit SÓ no treino; transform nos três conjuntos. Retorna arrays numpy float32."""
    Xtr = preprocessador.fit_transform(X_train)   # <- único lugar com fit
    Xva = preprocessador.transform(X_val)
    Xte = preprocessador.transform(X_test)
    return (
        np.asarray(Xtr, dtype=np.float32),
        np.asarray(Xva, dtype=np.float32),
        np.asarray(Xte, dtype=np.float32),
    )


def para_tensores(X, y, device):
    """Converte para tensores: float32 em X, long em y."""
    X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32).to(device)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long).to(device)
    return X_t, y_t


def resumo_preprocessamento(preprocessador):
    """Texto com as features finais (após one-hot) para registrar como artefato no MLflow."""
    nomes = list(preprocessador.get_feature_names_out())
    linhas = [
        "Resumo do pré-processamento (ajustado apenas no conjunto de TREINO)",
        "",
        f"Numéricos   ({len(COLUNAS_NUMERICAS)}): {COLUNAS_NUMERICAS} -> mediana + StandardScaler",
        f"Categóricos ({len(COLUNAS_CATEGORICAS)}): {COLUNAS_CATEGORICAS} -> moda + OneHotEncoder",
        "",
        f"Total de features após one-hot: {len(nomes)}",
        "",
        "Lista de features:",
    ] + [f"  {i:3d}. {n}" for i, n in enumerate(nomes)]
    return "\n".join(linhas)
