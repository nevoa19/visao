import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import N_CLASSES
from tabelas import numero_br

PASTA_FIG = "resultados/figuras"

AZUL = '#0275d8'
VERMELHO = '#d9534f'
CINZA = '#777777'

def salvar_figura(nome):
    if not os.path.exists(PASTA_FIG):
        os.makedirs(PASTA_FIG)
    plt.savefig(PASTA_FIG + "/" + nome + ".pdf", bbox_inches="tight")
    plt.savefig(PASTA_FIG + "/" + nome + ".png", dpi=200, bbox_inches="tight")
    plt.close()

def grafico_exemplos(X_bruto, y, por_classe=3):
    fig, eixos = plt.subplots(por_classe, N_CLASSES, figsize=(N_CLASSES * 0.8, por_classe * 0.85), dpi=150)
    rotulos = y.tolist()
    c = 0
    while c < N_CLASSES:
        achados = 0
        i = 0
        while i < len(rotulos) and achados < por_classe:
            if rotulos[i] == c:
                eixos[achados][c].imshow(X_bruto[i], cmap="gray_r")
                achados += 1
            i += 1
        lin = 0
        while lin < por_classe:
            eixos[lin][c].set_xticks([])
            eixos[lin][c].set_yticks([])
            lin += 1
        eixos[0][c].set_title(str(c), fontsize=9)
        c += 1
    salvar_figura("exemplos")

def grafico_distribuicao(cont_tr, cont_val, cont_te):
    plt.figure(figsize=(5.0, 2.4), dpi=150)
    x = []
    x_val = []
    x_te = []
    c = 0
    while c < N_CLASSES:
        x.append(c - 0.27)
        x_val.append(c)
        x_te.append(c + 0.27)
        c += 1
    plt.bar(x, cont_tr, width=0.27, color=AZUL, label="treino")
    plt.bar(x_val, cont_val, width=0.27, color=CINZA, label="validação")
    plt.bar(x_te, cont_te, width=0.27, color=VERMELHO, label="teste")
    plt.xticks(range(N_CLASSES), fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("dígito", fontsize=8)
    plt.ylabel("nº de imagens", fontsize=8)
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    salvar_figura("distribuicao_classes")

def grafico_perda(hist, titulo, nome):
    plt.figure(figsize=(3.4, 2.4), dpi=150)
    plt.plot(hist["epoca"], hist["perda_treino"], "o-", ms=3, color=AZUL, label="treino")
    plt.plot(hist["epoca"], hist["perda_val"], "s-", ms=3, color=VERMELHO, label="validação")
    plt.axvline(hist["melhor_epoca"], color=CINZA, ls="--", lw=0.8, label="menor perda val.")
    plt.xlabel("época", fontsize=8)
    plt.ylabel("entropia cruzada", fontsize=8)
    plt.title(titulo, fontsize=9)
    plt.tick_params(labelsize=7)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=7)
    salvar_figura(nome)

def grafico_busca(busca):
    plt.figure(figsize=(3.6, 2.6), dpi=150)
    for nome in busca:
        h = busca[nome]["hist"]
        plt.plot(h["epoca"], h["perda_val"], "o-", ms=2, lw=1, label=nome)
    plt.xlabel("época", fontsize=8)
    plt.ylabel("perda de validação", fontsize=8)
    plt.tick_params(labelsize=7)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=6)
    salvar_figura("mlp_busca")

def grafico_confusao(cm, titulo, nome):
    norm = []
    a = 0
    while a < N_CLASSES:
        total = 0
        for v in cm[a]:
            total += v
        linha = []
        for v in cm[a]:
            linha.append(v / total)
        norm.append(linha)
        a += 1

    plt.figure(figsize=(3.6, 3.4), dpi=150)
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=0.02)
    a = 0
    while a < N_CLASSES:
        b = 0
        while b < N_CLASSES:
            cor = "black"
            if a == b:
                cor = "white"
            if cm[a][b] > 0:
                plt.text(b, a, str(cm[a][b]), ha="center", va="center", color=cor, fontsize=5.5)
            b += 1
        a += 1
    plt.xticks(range(N_CLASSES), fontsize=7)
    plt.yticks(range(N_CLASSES), fontsize=7)
    plt.xlabel("previsto", fontsize=8)
    plt.ylabel("real", fontsize=8)
    plt.title(titulo, fontsize=9)
    salvar_figura(nome)

def grafico_erros(X_bruto, exemplos, nome, por_par=6):
    if len(exemplos) == 0:
        return
    fig, eixos = plt.subplots(len(exemplos), por_par, figsize=(por_par * 0.8, len(exemplos) * 0.9), dpi=150, squeeze=False)
    lin = 0
    while lin < len(exemplos):
        real = exemplos[lin][0]
        prev = exemplos[lin][1]
        idx = exemplos[lin][2]
        col = 0
        while col < por_par:
            eixo = eixos[lin][col]
            if col < len(idx):
                eixo.imshow(X_bruto[idx[col]], cmap="gray_r")
            eixo.set_xticks([])
            eixo.set_yticks([])
            col += 1
        eixos[lin][0].set_ylabel(str(real) + " → " + str(prev), fontsize=8)
        lin += 1
    salvar_figura(nome)

def grafico_comparacao(res):
    nomes = ["MLP", "LeNet"]
    params = [res["mlp"]["parametros"], res["cnn"]["parametros"]]
    tempos = [res["mlp"]["hist"]["tempo_total"], res["cnn"]["hist"]["tempo_total"]]
    accs = [res["mlp"]["teste"]["acuracia"] * 100, res["cnn"]["teste"]["acuracia"] * 100]
    cores = [AZUL, VERMELHO]

    fig, eixos = plt.subplots(1, 3, figsize=(6.4, 2.0), dpi=150)
    valores = [params, tempos, accs]
    titulos = ["parâmetros", "tempo de treino (s)", "acurácia no teste (%)"]
    j = 0
    while j < 3:
        eixos[j].bar(nomes, valores[j], color=cores)
        eixos[j].set_title(titulos[j], fontsize=8)
        eixos[j].tick_params(labelsize=7)
        i = 0
        while i < 2:
            v = valores[j][i]
            if j == 0:
                texto = numero_br(int(v))
            elif j == 1:
                texto = numero_br(v, 1)
            else:
                texto = numero_br(v, 2)
            eixos[j].text(i, v, texto, ha="center", va="bottom", fontsize=6)
            i += 1
        j += 1
    menor = min(accs) - 1
    eixos[2].set_ylim(menor, 100)
    fig.tight_layout()
    salvar_figura("comparacao")
