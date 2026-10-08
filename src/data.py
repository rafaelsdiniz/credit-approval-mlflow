"""
Carregamento, limpeza e divisão do dataset Credit Approval (UCI, id=27).

Por quê: no código da aula os dados vinham prontos do scikit-learn (Iris).
Aqui o dataset vem do UCI, tem valores faltantes ('?') e colunas categóricas,
então precisamos de uma etapa explícita de carga e limpeza.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Nomes das colunas conforme a documentação do UCI (atributos anonimizados)
COLUNAS_ATRIBUTOS = [f"A{i}" for i in range(1, 16)]   # A1 ... A15
COLUNA_ALVO = "A16"                                   # '+' aprovado, '-' rejeitado

COLUNAS_NUMERICAS = ["A2", "A3", "A8", "A11", "A14", "A15"]
COLUNAS_CATEGORICAS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]


def carregar_dados(caminho_local, uci_id=27):
    """
    Retorna (DataFrame com as colunas A1..A16, texto descrevendo a origem).
    Na primeira execução baixa do UCI (via ucimlrepo) e salva uma cópia local;
    nas seguintes lê o arquivo local (mais rápido e funciona sem internet).
    """
    if os.path.exists(caminho_local):
        origem = "arquivo local (data/crx.data)"
        df = pd.read_csv(caminho_local, na_values=["?"])
    else:
        origem = "download UCI via ucimlrepo (id=27)"
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=uci_id)
        # Junta atributos (X) e alvo (y) em um único DataFrame
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        # A ordem das colunas pode não vir como A1..A16, então reordenamos pelo nome
        df = df[COLUNAS_ATRIBUTOS + [COLUNA_ALVO]]
        os.makedirs(os.path.dirname(caminho_local) or ".", exist_ok=True)
        df.to_csv(caminho_local, index=False)

    df = limpar_dados(df)
    return df, origem


def limpar_dados(df):
    """Padroniza tipos: numéricos viram float (com NaN onde havia '?') e o alvo vira 0/1."""
    df = df.copy()
    # Alguns numéricos (A2, A14) chegam como texto por causa dos '?'.
    # errors='coerce' transforma o que não é número em NaN (que depois será imputado).
    for col in COLUNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in COLUNAS_CATEGORICAS:
        df[col] = df[col].astype("object")
    # Alvo: '+' (aprovado) -> 1, '-' (rejeitado) -> 0
    df[COLUNA_ALVO] = (df[COLUNA_ALVO].astype(str).str.strip() == "+").astype(int)
    return df


def dividir_dados(df, frac_val, frac_test, seed):
    """
    Divide em treino / validação / teste de forma ESTRATIFICADA (mesma proporção de
    aprovados/rejeitados em cada parte). Duas chamadas de train_test_split, como no código da aula.

    Por quê estratificar: o dataset é pequeno (690 linhas) e levemente desbalanceado;
    sem estratificação um conjunto poderia ficar com proporção diferente dos outros.
    """
    X = df[COLUNAS_ATRIBUTOS]
    y = df[COLUNA_ALVO]

    # 1º corte: separa (val + teste) do treino
    frac_temp = frac_val + frac_test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=frac_temp, random_state=seed, stratify=y
    )
    # 2º corte: divide a parte temporária entre validação e teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=frac_test / frac_temp, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
