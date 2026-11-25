# Visão computacional com redes neurais

Mini-projeto 03 (IA): reconhecimento de dígitos manuscritos (MNIST) com uma rede totalmente conectada (MLP com saída softmax) e uma rede convolucional no estilo LeNet, em PyTorch.

## Requisitos
- Python 3.10 ou superior
- torch, torchvision, numpy, matplotlib

    pip install -r requirements.txt

## Como executar
    python visao.py

Na primeira execução o MNIST é baixado para `data/`. Usa GPU (CUDA ou MPS) se tiver; senão, CPU.
Em CPU o experimento inteiro leva alguns minutos (a busca do MLP é a parte mais demorada).
Para pular a busca e usar uma arquitetura fixa, alterar `RODAR_BUSCA_MLP` para `False` e definir `ARQ_MLP_FIXA` no início de `visao.py`.

## Arquivos
- `visao.py`: roda o experimento inteiro e gera tabelas e figuras
- `dados.py`: download do MNIST, média e desvio dos pixels, divisão treino/validação, normalização e contagem por classe
- `modelos.py`: MLP configurável, LeNet, inicialização e contagem de parâmetros
- `treino.py`: laço de treino com SGD, perda por época e avaliação
- `metricas.py`: matriz de confusão, acurácia, precisão, revocação e F1 macro e pares mais confundidos
- `graficos.py`: figuras
- `tabelas.py`: tabelas em LaTeX

## Protocolo
- validação: 6.000 imagens sorteadas do treino (semente 42); treino com as 54.000 restantes
- normalização: (x − média) / desvio, com média e desvio calculados nas 60.000 imagens de treino
- perda: entropia cruzada (`CrossEntropyLoss`, que já aplica o softmax nos logits)
- MLP: ReLU, SGD (η = 0,1, lote 256), inicialização Kaiming; busca entre 7 arquiteturas (1 a 3 camadas ocultas) por 10 épocas, escolhendo a de maior acurácia de validação (empate: menos parâmetros); a escolhida é treinada do zero por 20 épocas
- LeNet: conv 5×5 (6) → max pool 2×2 → conv 5×5 (16) → max pool 2×2 → 400 → 120 → 84 → 10, ReLU, SGD (η = 0,05, lote 256), inicialização Kaiming, 15 épocas
- em cada treino ficam os pesos da época com menor perda de validação
- o teste é usado uma única vez, no final de cada modelo
- precisão, revocação e F1 são macro (média simples das 10 classes)

## Saídas
Em `resultados/`:
- `tabelas/`: `dados_classes`, `mlp_busca`, `metricas_teste`, `comparacao`, `pares_confundidos`, `f1_classes`, `lenet_parametros`
- `figuras/`: exemplos de dígitos, distribuição por classe, busca do MLP, curvas de perda, matrizes de confusão, exemplos dos pares mais confundidos e comparação MLP x LeNet
- `resultados.json`: todos os números

## Referência
A. Zhang, Z. C. Lipton, M. Li e A. J. Smola. *Dive into Deep Learning*. d2l.ai.
