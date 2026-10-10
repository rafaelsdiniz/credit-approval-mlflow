"""MLP (Linear -> ReLU -> Linear) como na aula, mas com as camadas ocultas vindas da config."""

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
                camadas.append(nn.Dropout(dropout))
            tamanho_anterior = n_neuronios
        # Saída: 2 logits; a CrossEntropyLoss aplica o softmax
        camadas.append(nn.Linear(tamanho_anterior, n_classes))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)
