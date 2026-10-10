"""Leitura do YAML do experimento e sobrescrita de chaves via `--set chave.sub=valor`."""

import yaml


def carregar_config(caminho_yaml, sobrescritas=None):
    """Lê o YAML e aplica as sobrescritas do tipo 'treino.lr=0.001'."""
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for item in sobrescritas or []:
        if "=" not in item:
            raise ValueError(f"Formato inválido em --set: '{item}'. Use chave.sub=valor")
        chave, valor_texto = item.split("=", 1)
        valor = yaml.safe_load(valor_texto)  # "0.3" -> float, "[128,64]" -> lista, "true" -> bool
        _definir_chave(config, chave.split("."), valor)

    return _corrigir_numeros(config)


def _corrigir_numeros(valor):
    """Converte strings numéricas ("1e-3") para float: o PyYAML só entende "1.0e-3", e a string quebrava o Adam."""
    if isinstance(valor, dict):
        return {k: _corrigir_numeros(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_corrigir_numeros(v) for v in valor]
    if isinstance(valor, str):
        try:
            return float(valor)
        except ValueError:
            return valor
    return valor


def _definir_chave(dicionario, partes, valor):
    """Navega pelas chaves aninhadas (ex.: ['treino', 'lr']) e define o valor."""
    for parte in partes[:-1]:
        dicionario = dicionario.setdefault(parte, {})
    dicionario[partes[-1]] = valor


def achatar_config(config, prefixo=""):
    """Achata o dicionário aninhado em chaves 'a.b.c', formato que o MLflow aceita como parâmetro."""
    achatado = {}
    for chave, valor in config.items():
        nome = f"{prefixo}{chave}"
        if isinstance(valor, dict):
            achatado.update(achatar_config(valor, prefixo=f"{nome}."))
        else:
            achatado[nome] = valor
    return achatado
