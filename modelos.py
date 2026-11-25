from torch import nn

ARQUITETURAS_MLP = [
    [128],
    [256],
    [512],
    [256, 128],
    [512, 256],
    [256, 128, 64],
    [512, 256, 128],
]

def nome_arquitetura(ocultas):
    texto = ""
    for n in ocultas:
        if texto != "":
            texto = texto + "x"
        texto = texto + str(n)
    return texto

class MLP(nn.Module):
    def __init__(self, ocultas, entrada=784, saida=10):
        super().__init__()
        camadas = [nn.Flatten()]
        anterior = entrada
        for n in ocultas:
            camadas.append(nn.Linear(anterior, n))
            camadas.append(nn.ReLU())
            anterior = n
        camadas.append(nn.Linear(anterior, saida))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

class LeNet(nn.Module):
    def __init__(self, saida=10):
        super().__init__()
        camadas = []
        camadas.append(nn.Conv2d(1, 6, kernel_size=5, padding=2))
        camadas.append(nn.ReLU())
        camadas.append(nn.MaxPool2d(2, 2))
        camadas.append(nn.Conv2d(6, 16, kernel_size=5))
        camadas.append(nn.ReLU())
        camadas.append(nn.MaxPool2d(2, 2))
        camadas.append(nn.Flatten())
        camadas.append(nn.Linear(16 * 5 * 5, 120))
        camadas.append(nn.ReLU())
        camadas.append(nn.Linear(120, 84))
        camadas.append(nn.ReLU())
        camadas.append(nn.Linear(84, saida))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

def iniciar_pesos(modelo):
    for m in modelo.modules():
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

def contar_parametros(modelo):
    total = 0
    for p in modelo.parameters():
        if p.requires_grad:
            total += p.numel()
    return total

def parametros_por_camada(modelo):
    linhas = []
    for nome, camada in modelo.rede.named_children():
        n = 0
        for p in camada.parameters():
            n += p.numel()
        if n > 0:
            linhas.append([nome, type(camada).__name__, n])
    return linhas
