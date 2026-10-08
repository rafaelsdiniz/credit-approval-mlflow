"""
Definição da rede neural (MLP) em PyTorch.

É a mesma ideia da classe IrisModel do código da aula (Linear -> ReLU -> Linear),
só que o número e o tamanho das camadas ocultas vêm da configuração.
Ex.: camadas_ocultas=[16] reproduz a arquitetura da aula; [128, 64] cria uma rede maior.
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_entradas, camadas_ocultas, n_classes=2, dropout=0.0):
        super().__init__()
        camadas = []
        tamanho_anterior = n_entradas
        for n_neuronios in camadas_ocultas:
            camadas.append(nn.Linear(tamanho_anterior, n_neuronios))
            camadas.append(nn.ReLU())
            if dropout > 0:
                # Dropout desliga neurônios aleatoriamente no treino: ajuda a reduzir overfitting
                camadas.append(nn.Dropout(dropout))
            tamanho_anterior = n_neuronios
        # Camada de saída: 2 logits (rejeitado / aprovado).
        # A CrossEntropyLoss aplica o softmax internamente, igual ao código da aula.
        camadas.append(nn.Linear(tamanho_anterior, n_classes))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)
