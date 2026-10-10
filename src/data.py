"""Carregamento, limpeza e divisão do dataset Credit Approval (UCI, id=27)."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

COLUNAS_ATRIBUTOS = [f"A{i}" for i in range(1, 16)]   # A1 ... A15 (anonimizados)
COLUNA_ALVO = "A16"                                   # '+' aprovado, '-' rejeitado

COLUNAS_NUMERICAS = ["A2", "A3", "A8", "A11", "A14", "A15"]
COLUNAS_CATEGORICAS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]


def carregar_dados(caminho_local, uci_id=27):
    """Retorna (DataFrame A1..A16, origem). Baixa do UCI na 1ª vez e salva a cópia local."""
    if os.path.exists(caminho_local):
        origem = "arquivo local (data/crx.data)"
        df = pd.read_csv(caminho_local, na_values=["?"])
    else:
        origem = "download UCI via ucimlrepo (id=27)"
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=uci_id)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df = df[COLUNAS_ATRIBUTOS + [COLUNA_ALVO]]  # garante a ordem A1..A16
        os.makedirs(os.path.dirname(caminho_local) or ".", exist_ok=True)
        df.to_csv(caminho_local, index=False)

    df = limpar_dados(df)
    return df, origem


def limpar_dados(df):
    """Padroniza tipos: numéricos viram float (com NaN onde havia '?') e o alvo vira 0/1."""
    df = df.copy()
    for col in COLUNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # '?' vira NaN (imputado depois)
    for col in COLUNAS_CATEGORICAS:
        df[col] = df[col].astype("object")
    df[COLUNA_ALVO] = (df[COLUNA_ALVO].astype(str).str.strip() == "+").astype(int)
    return df


def dividir_dados(df, frac_val, frac_test, seed, estratificado=True):
    """
    Divide em treino / validação / teste, estratificado (mesma proporção de classes em cada parte).
    random_state=seed garante o mesmo split em todas as rodadas: só a configuração muda, os dados não.
    """
    X = df[COLUNAS_ATRIBUTOS]
    y = df[COLUNA_ALVO]

    frac_temp = frac_val + frac_test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=frac_temp, random_state=seed, stratify=y if estratificado else None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=frac_test / frac_temp, random_state=seed,
        stratify=y_temp if estratificado else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def descrever_divisao(X_train, X_val, X_test, y_train, y_val, y_test):
    """CSV com índice, conjunto e classe de cada amostra: artefato que prova que o split é igual nas rodadas."""
    partes = [
        pd.DataFrame({"indice": X_train.index, "conjunto": "treino", "classe": y_train.values}),
        pd.DataFrame({"indice": X_val.index, "conjunto": "validacao", "classe": y_val.values}),
        pd.DataFrame({"indice": X_test.index, "conjunto": "teste", "classe": y_test.values}),
    ]
    tabela = pd.concat(partes).sort_values("indice")
    resumo = tabela.groupby("conjunto")["classe"].agg(["count", "mean"]).rename(
        columns={"count": "n", "mean": "proporcao_aprovados"}
    )
    cabecalho = "# Divisão estratificada com seed fixa. Resumo por conjunto:\n"
    cabecalho += "".join(f"#   {c}: n={int(r.n)}, aprovados={r.proporcao_aprovados:.3f}\n" for c, r in resumo.iterrows())
    return cabecalho + tabela.to_csv(index=False)
