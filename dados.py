import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets

SEMENTE = 42
PASTA_DADOS = "data"
N_VALIDACAO = 6000
N_CLASSES = 10

def baixar_mnist():
    treino = datasets.MNIST(PASTA_DADOS, train=True, download=True)
    teste = datasets.MNIST(PASTA_DADOS, train=False, download=True)
    X_treino = treino.data.float() / 255.0
    y_treino = treino.targets.clone()
    X_teste = teste.data.float() / 255.0
    y_teste = teste.targets.clone()
    return X_treino, y_treino, X_teste, y_teste

def media_desvio(X):
    media = X.mean().item()
    desvio = X.std().item()
    return media, desvio

def dividir_validacao(X, y):
    gerador = torch.Generator()
    gerador.manual_seed(SEMENTE)
    ordem = torch.randperm(len(X), generator=gerador)
    idx_val = ordem[:N_VALIDACAO]
    idx_tr = ordem[N_VALIDACAO:]
    return X[idx_tr], y[idx_tr], X[idx_val], y[idx_val]

def normalizar(X, media, desvio):
    Xn = (X - media) / desvio
    return Xn.unsqueeze(1)

def contar_por_classe(y):
    contagem = []
    c = 0
    while c < N_CLASSES:
        contagem.append(0)
        c += 1
    for rotulo in y.tolist():
        contagem[rotulo] += 1
    return contagem

def preparar_dados():
    X_treino_todo, y_treino_todo, X_teste, y_teste = baixar_mnist()
    media, desvio = media_desvio(X_treino_todo)
    X_tr, y_tr, X_val, y_val = dividir_validacao(X_treino_todo, y_treino_todo)

    d = {}
    d["media"] = media
    d["desvio"] = desvio
    d["X_tr"] = normalizar(X_tr, media, desvio)
    d["y_tr"] = y_tr
    d["X_val"] = normalizar(X_val, media, desvio)
    d["y_val"] = y_val
    d["X_te"] = normalizar(X_teste, media, desvio)
    d["y_te"] = y_teste
    d["X_tr_bruto"] = X_tr
    d["X_te_bruto"] = X_teste
    return d

def criar_loader(X, y, lote, embaralhar):
    gerador = torch.Generator()
    gerador.manual_seed(SEMENTE)
    return DataLoader(TensorDataset(X, y), batch_size=lote, shuffle=embaralhar, generator=gerador)
