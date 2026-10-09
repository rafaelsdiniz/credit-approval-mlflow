# Pipeline de treino e avaliação com MLflow — Credit Approval

**Disciplina:** Inteligência Artificial 2026/2 — Sistemas de Informação / UNITINS
**Avaliação:** A1 — "Treinamento observável: como rastrear, comparar e reproduzir modelos"
**Aluno:** Rafael Silva Diniz
**Vídeo de apresentação:** _(link do YouTube aqui)_

Este projeto pega o script da aula [`mlp_torch_avaliacao.py`](https://github.com/sousamaf/AI-Lab/blob/main/algorithms/neural_networks/mlp/mlp_torch_avaliacao.py)
(uma MLP em PyTorch para o Iris) e o reorganiza como um **pipeline de treino → validação → teste**,
configurável por arquivo YAML e com **rastreamento completo no MLflow** (parâmetros, métricas por época,
artefatos, modelo e trace das etapas), usando o dataset **Credit Approval**.

---

## 1. O problema e o dataset

O **Credit Approval** (UCI, id 27) tem 690 pedidos de cartão de crédito. Cada linha tem 15 atributos
anonimizados (A1 a A15) e o alvo A16: `+` = pedido **aprovado** (classe 1) e `-` = pedido **rejeitado** (classe 0).
É um problema de **classificação binária**.

| Tipo | Colunas | Tratamento |
|---|---|---|
| Numéricas | A2, A3, A8, A11, A14, A15 | imputação pela **mediana** + `StandardScaler` |
| Categóricas | A1, A4, A5, A6, A7, A9, A10, A12, A13 | imputação pela **moda** + `OneHotEncoder(handle_unknown="ignore")` |

Valores faltantes vêm marcados com `?` (em A1, A2, A4, A5, A6, A7 e A14). Depois do one-hot, as 15 colunas
viram **46 features**. A divisão é **estratificada**: 70 % treino (483), 15 % validação (103), 15 % teste (104).

O dataset é baixado uma única vez com `ucimlrepo` e salvo em `data/crx.data`; as execuções seguintes usam
o arquivo local (funciona offline).

## 2. Perguntas dos experimentos

Cada arquivo em `configs/` é um experimento do MLflow e responde a uma pergunta objetiva:

| Experimento | Config | Pergunta |
|---|---|---|
| `exp01_mlp_pequena` | `configs/exp01_mlp_pequena.yaml` | Uma MLP pequena (1 camada oculta com 16 neurônios, como no código da aula) já consegue boa F1 para aprovar/rejeitar crédito? |
| `exp02_mlp_grande` | `configs/exp02_mlp_grande.yaml` | Uma MLP maior (128-64) melhora o resultado ou só aumenta o overfitting num dataset pequeno de 690 linhas? |

Dentro do `exp02` há ainda duas runs de **regularização** criadas por linha de comando (`--set`), sem editar
nenhum arquivo: `dropout=0.3` e `weight_decay=1e-3`. Elas investigam se a regularização reduz o overfitting
observado na MLP grande.

## 3. Estrutura de pastas

```
credit-approval-mlflow/
├── configs/
│   ├── exp01_mlp_pequena.yaml   # MLP [16]        (experimento 1)
│   └── exp02_mlp_grande.yaml    # MLP [128, 64]   (experimento 2)
├── data/
│   └── crx.data                 # cópia local do Credit Approval (baixada do UCI)
├── src/
│   ├── config.py                # lê o YAML e aplica --set chave.sub=valor
│   ├── data.py                  # carrega, limpa e divide (treino/val/teste estratificado)
│   ├── features.py              # ColumnTransformer ajustado SÓ no treino
│   ├── model.py                 # classe MLP (camadas ocultas configuráveis)
│   ├── train.py                 # pipeline principal + MLflow (ponto de entrada)
│   └── comparar.py              # imprime a tabela comparativa das runs
├── run_all.ps1 / run_all.sh     # roda todos os experimentos em sequência
├── environment.yml              # ambiente conda (python 3.11 + pip -r requirements.txt)
├── requirements.txt
└── README.md
```

Gerados ao executar (ignorados pelo git): `mlflow.db` (banco do MLflow) e `mlruns/` (artefatos).

## 4. Instalação

Requisitos: Python 3.11 (ou conda). O `torch` em versão CPU é suficiente.

**Opção A — venv (Windows / PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Opção B — Linux / macOS (script pronto)**

```bash
bash setup.sh                 # cria o .venv e instala tudo (torch CPU + requirements)
source .venv/bin/activate
```

**Opção C — conda**

```bash
conda env create -f environment.yml
conda activate credit-approval-mlflow
```

(no Linux/macOS, use `source .venv/bin/activate` em vez do `Activate.ps1`).

## 5. Como rodar

Um experimento por comando, sempre a partir da raiz do projeto:

```bash
python -m src.train --config configs/exp01_mlp_pequena.yaml
python -m src.train --config configs/exp02_mlp_grande.yaml
```

Qualquer chave do YAML pode ser sobrescrita com `--set` (pode repetir):

```bash
python -m src.train --config configs/exp02_mlp_grande.yaml --set modelo.dropout=0.3 --set experimento.run_name=mlp_128_64_dropout03
python -m src.train --config configs/exp02_mlp_grande.yaml --set treino.weight_decay=1e-3 --set experimento.run_name=mlp_128_64_wd1e-3
python -m src.train --config configs/exp01_mlp_pequena.yaml --set treino.lr=0.001 --set modelo.camadas_ocultas=[32,16]
```

Para rodar tudo de uma vez: `.\run_all.ps1` (Windows) ou `bash run_all.sh`.
Para ver a tabela comparativa no terminal: `python -m src.comparar`.

### Abrir o MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Depois acesse <http://127.0.0.1:5000>. Na aba **Experiments** selecione as runs e clique em **Compare**
para ver parâmetros e curvas lado a lado; na aba **Traces** de cada experimento estão as etapas do pipeline.

## 6. O que é registrado em cada run

| Categoria | Conteúdo |
|---|---|
| **Tags** | pergunta do experimento, dataset, origem dos dados (download ou arquivo local), device, código-base |
| **Parâmetros** | toda a config achatada (`modelo.camadas_ocultas`, `treino.lr`, `treino.batch_size`, `seed`, `dados.frac_val`...) + `n_train`, `n_val`, `n_test`, `n_features` |
| **Métricas por época** (`step` = época) | `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` |
| **Métricas finais** | validação: `val_accuracy_final`, `val_precision`, `val_recall`, `val_f1`, `val_roc_auc`; teste: `test_loss`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`; e `melhor_epoca`, `epocas_executadas` |
| **Artefatos** | `curvas_treinamento.png` (perda e acurácia, com a melhor época marcada), `matriz_confusao_teste.png`, `classification_report_teste.txt`, `config_usada.yaml`, `resumo_preprocessamento.txt`, `preprocessador.joblib`, e o modelo (`mlflow.pytorch.log_model`) |
| **Trace** | um trace por run com os spans `preparar_dados`, `treinar` e `validar_e_testar` (entradas, saídas, duração e erro, se houver) |

Precision, recall e F1 usam `average="macro"` (média das duas classes), como no script original.

## 7. Como o pipeline funciona (passo a passo)

1. **Configurar** — `src/config.py` lê o YAML e aplica os `--set`. Nada de hiperparâmetro fixo no código.
2. **Preparar** (span `preparar_dados`) — `src/data.py` carrega o dataset, converte o alvo para 0/1 e divide em
   treino/validação/teste de forma estratificada. `src/features.py` **ajusta** o pré-processador no treino e
   **aplica** nos três conjuntos. O treino vai para um `DataLoader` com mini-batches embaralhados.
3. **Treinar** (span `treinar`) — mesma receita da aula: `nn.Module` → `CrossEntropyLoss` → `Adam`; a cada época
   um passo por mini-batch, depois a validação sem gradiente. As quatro métricas são registradas por época.
   **Early stopping**: se a `val_loss` não melhora por `paciencia` épocas, para; os pesos da **melhor época**
   são guardados e restaurados.
4. **Validar e testar** (span `validar_e_testar`) — métricas finais na validação (base da comparação entre
   runs) e, **uma única vez**, no teste, com o modelo da melhor época. Gera as figuras e salva os artefatos.

A **seed** fixa numpy, torch (inclusive o embaralhamento do `DataLoader`) e o `random_state` do split, então
rodar o mesmo comando duas vezes produz exatamente os mesmos números (foi verificado).

## 8. Resultados (números reais, seed 42)

| Experimento | Run | Camadas | Dropout | L2 | Melhor época | val_f1 | val_roc_auc | test_accuracy | test_f1 | test_roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| exp01_mlp_pequena | `mlp_16` | [16] | 0 | 0 | 10 | **0,9013** | **0,9722** | **0,8750** | **0,8736** | 0,9520 |
| exp02_mlp_grande | `mlp_128_64` | [128, 64] | 0 | 0 | 4 | 0,9214 | 0,9641 | 0,8365 | 0,8362 | 0,9543 |
| exp02_mlp_grande | `mlp_128_64_dropout03` | [128, 64] | 0,3 | 0 | 11 | 0,9211 | 0,9691 | 0,8558 | 0,8547 | 0,9397 |
| exp02_mlp_grande | `mlp_128_64_wd1e-3` | [128, 64] | 0 | 1e-3 | 4 | 0,9114 | 0,9638 | 0,8558 | 0,8554 | 0,9550 |

Todas as runs usam lr = 0,01, batch 32, máximo de 200 épocas e paciência 20, e o mesmo split.
A tabela é gerada por `python -m src.comparar`.

Matriz de confusão do modelo escolhido (`mlp_16`) no teste: 51 rejeitados corretos, 40 aprovados corretos,
7 rejeitados previstos como aprovados e 6 aprovados previstos como rejeitados (104 amostras).

### O que as curvas mostram

* **`mlp_16`** — a `val_loss` cai até a época 10 (0,23) e depois sobe devagar (0,31 na época 30), enquanto a
  `train_loss` continua caindo (0,15). Overfitting **leve** e tardio; o early stopping parou na época 30.
* **`mlp_128_64`** — a `val_loss` atinge o mínimo já na **época 4** (0,24) e dispara até 0,75 na época 24,
  enquanto a `train_loss` cai para 0,05 e a acurácia de treino chega a 98 %. A acurácia de validação cai de
  0,92 para 0,82. É o quadro clássico de **overfitting forte**: a rede decora o treino.
* **`mlp_128_64_dropout03`** — o dropout adia o overfitting (melhor época 4 → 11) e a `val_loss` sobe mais
  devagar, mas ainda sobe. No teste melhora a F1 de 0,836 para 0,855.
* **`mlp_128_64_wd1e-3`** — o weight decay não adia a melhor época (continua 4), mas suaviza o crescimento da
  `val_loss` e também leva a F1 de teste a 0,855.

## 9. Conclusão

**Pergunta 1 — a MLP pequena já resolve?** Sim. Com apenas uma camada oculta de 16 neurônios (a mesma
arquitetura da aula) a rede atinge F1 macro de 0,90 na validação e **0,87 no teste**, com ROC-AUC de 0,95.
Para um dataset com 690 linhas e 46 features, essa capacidade é suficiente.

**Pergunta 2 — a MLP maior melhora ou só aumenta o overfitting?** Só aumenta o overfitting. A rede 128-64 tem
cerca de 14 mil parâmetros para 483 exemplos de treino: ela chega à melhor validação em 4 épocas e a partir daí
a `val_loss` sobe enquanto a `train_loss` cai, que é a assinatura do overfitting. A F1 de validação foi
levemente maior (0,921 contra 0,901, diferença de cerca de 2 amostras em 103), mas no teste ficou **pior**
(0,836 contra 0,874). A regularização (dropout ou L2) reduz o dano, mas não faz a rede grande superar a pequena.

**Modelo escolhido: `mlp_16` (exp01).** A escolha foi feita na validação, antes de olhar o teste: as duas
arquiteturas empatam em F1 de validação dentro da margem de erro, mas a pequena tem a maior `val_roc_auc`
(0,972), a curva de validação mais estável e cerca de 18 vezes menos parâmetros (786 contra 14.402). O teste, usado uma única vez, confirmou
a escolha.

**Ressalva importante:** validação e teste têm ~100 amostras cada, então **1 amostra ≈ 1 ponto percentual**.
Diferenças de 2 a 4 pontos entre runs (como 0,855 contra 0,874) são inconclusivas; o que sustenta a decisão não
é só o número final, mas o **comportamento das curvas** registradas no MLflow.

## 10. Diagnóstico de um incidente (run com falha)

Ao criar a run de weight decay com `--set treino.weight_decay=1e-3`, a run `mlp_128_64_wd1e-3` **falhou**
(status `FAILED` no MLflow). O trace mostra o span `preparar_dados` concluído e o erro no span `treinar`:

```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

* **Sintoma:** falha no `optim.Adam(...)`, antes da primeira época.
* **Hipótese:** o valor `1e-3` chegou como texto. O PyYAML só reconhece notação científica com ponto
  (`1.0e-3`); `1e-3` vira a string `'1e-3'`. Os parâmetros registrados na run confirmam
  (`treino.weight_decay = 1e-3` como string, enquanto nas outras runs é `0.0`).
* **Correção:** `src/config.py` passou a converter strings numéricas para `float` (função `_corrigir_numeros`).
  O commit da correção está no histórico do git.
* **Nova run:** o mesmo comando rodou com sucesso e gerou a linha `mlp_128_64_wd1e-3` da tabela.

A run com falha foi mantida no MLflow de propósito, como evidência do diagnóstico.

## 11. Correção do vazamento de dados em relação ao código original

No script da aula, o `StandardScaler` é ajustado em **todos** os dados e só depois vem o `train_test_split`:

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)            # usa média e desvio de TODAS as linhas
X_train, X_temp, ... = train_test_split(X, y, ...)
```

Assim, a média e o desvio usados para padronizar o treino já "viram" as linhas de validação e teste.
Isso é **vazamento de informação (data leakage)**: a avaliação fica otimista, porque o teste deixou de ser
dado realmente novo.

Aqui a ordem é invertida: primeiro o split, depois o pré-processador é ajustado **somente no treino**
(`fit_transform(X_train)`) e apenas aplicado nos outros conjuntos (`transform(X_val)`, `transform(X_test)`),
em `src/features.py`. O mesmo vale para a imputação (mediana e moda calculadas no treino) e para o one-hot
(categorias aprendidas no treino; categorias desconhecidas viram zeros graças ao `handle_unknown="ignore"`).
O pré-processador ajustado é salvo como artefato (`preprocessador.joblib`), para que novos dados sejam
transformados exatamente como o treino.

## 12. Outras diferenças em relação ao script original

| Script da aula | Este projeto | Por quê |
|---|---|---|
| Iris (3 classes, 4 features) | Credit Approval (2 classes, 46 features após one-hot) | dataset atribuído |
| scaler em todos os dados | pré-processador ajustado só no treino | evita vazamento |
| split sem estratificação, sem seed global | split estratificado + seeds fixas | reprodutibilidade |
| full-batch, 1000 épocas fixas | mini-batches (`DataLoader`) + early stopping pela `val_loss` | para na melhor época |
| hiperparâmetros no código | YAML + `--set` | repetir sem editar o script |
| `plt.show()` | figuras salvas e registradas no MLflow (backend `Agg`) | roda sem janela, fica no histórico |
| métricas só no terminal | MLflow: params, métricas por época, artefatos, modelo, trace | comparar e reproduzir |

O que **não** mudou: `nn.Module` com `Linear → ReLU → Linear`, `CrossEntropyLoss`, `Adam`, loop manual de
treino/validação e as métricas precision/recall/F1/matriz de confusão.

## 13. Uso de IA generativa

O uso de IA generativa é obrigatório na disciplina. Este projeto foi desenvolvido com o **Claude** (Anthropic),
usado em modo agente (Claude Code) para: ler o script original e propor o plano de reorganização; escrever os
módulos e os comentários explicativos; instalar o ambiente e executar os experimentos; diagnosticar o incidente
do `--set` e a colisão de IDs de trace causada por `random.seed`; e redigir este README a partir dos
resultados reais. As decisões de projeto (dataset, perguntas, arquiteturas, métricas, escolha do modelo) e a
revisão de todo o código são de responsabilidade do aluno, que apresenta e explica o projeto no vídeo.

## 14. Referências

* Script original: `sousamaf/AI-Lab` — `algorithms/neural_networks/mlp/mlp_torch_avaliacao.py`
* Dataset: Quinlan, J. R. *Credit Approval*. UCI Machine Learning Repository, id 27.
* MLflow Tracking e MLflow Tracing — documentação oficial (<https://mlflow.org/docs/latest/>)
* PyTorch — <https://pytorch.org/docs/>
