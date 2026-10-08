"""
Leitura da configuração do experimento (arquivo YAML) e sobrescrita via linha de comando.

Por quê: a ideia central do trabalho é não editar o script a cada tentativa.
Toda escolha (arquitetura, lr, batch, épocas...) fica em um YAML, e qualquer
chave pode ser trocada com `--set chave.sub=valor` sem mexer no arquivo.
"""

import yaml


def carregar_config(caminho_yaml, sobrescritas=None):
    """Lê o YAML e aplica as sobrescritas do tipo 'treino.lr=0.001'."""
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for item in sobrescritas or []:
        if "=" not in item:
            raise ValueError(f"Formato inválido em --set: '{item}'. Use chave.sub=valor")
        chave, valor_texto = item.split("=", 1)
        # yaml.safe_load converte o texto para o tipo certo:
        # "0.3" -> float, "[128,64]" -> lista, "true" -> bool, "abc" -> str
        valor = yaml.safe_load(valor_texto)
        _definir_chave(config, chave.split("."), valor)

    return _corrigir_numeros(config)


def _corrigir_numeros(valor):
    """
    Converte strings que na verdade são números (ex.: "1e-3") para float.
    Por quê: o PyYAML só reconhece notação científica com ponto ("1.0e-3");
    "1e-3" vira a STRING '1e-3' e o Adam quebra ao comparar weight_decay com 0.0.
    (Esse foi o incidente registrado na run com falha; ver README.)
    """
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
    """
    Transforma o dicionário aninhado em chaves 'a.b.c' -> valor.
    Por quê: o MLflow registra parâmetros como pares chave/valor simples,
    então {'treino': {'lr': 0.01}} vira {'treino.lr': 0.01}.
    """
    achatado = {}
    for chave, valor in config.items():
        nome = f"{prefixo}{chave}"
        if isinstance(valor, dict):
            achatado.update(achatar_config(valor, prefixo=f"{nome}."))
        else:
            achatado[nome] = valor
    return achatado
