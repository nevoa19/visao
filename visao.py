import os
import json
import torch
from dados import SEMENTE, N_CLASSES, preparar_dados, criar_loader, contar_por_classe
from modelos import ARQUITETURAS_MLP, MLP, LeNet, nome_arquitetura, iniciar_pesos, contar_parametros, parametros_por_camada
from treino import escolher_dispositivo, treinar, avaliar
from metricas import calcular_metricas, pares_mais_confundidos, indices_erros
from tabelas import numero_br, salvar_tabela
from graficos import grafico_exemplos, grafico_distribuicao, grafico_perda, grafico_busca, grafico_confusao, grafico_erros, grafico_comparacao

RODAR_BUSCA_MLP = True 
ARQ_MLP_FIXA = [512, 256]

EPOCAS_BUSCA = 10
EPOCAS_MLP = 20
LOTE_MLP = 256
LR_MLP = 0.1
MOMENTO_MLP = 0.0 

EPOCAS_CNN = 15
LOTE_CNN = 256
LR_CNN = 0.05 
MOMENTO_CNN = 0.0

def fixar_semente():
    torch.manual_seed(SEMENTE)

def descrever_dados(d):
    print("\n=== Dados")
    cont_tr = contar_por_classe(d["y_tr"])
    cont_val = contar_por_classe(d["y_val"])
    cont_te = contar_por_classe(d["y_te"])
    print(f"treino={len(d['y_tr'])}  validação={len(d['y_val'])}  teste={len(d['y_te'])}")
    print(f"média={d['media']:.4f}  desvio={d['desvio']:.4f}")

    cab = ["classe"]
    c = 0
    while c < N_CLASSES:
        cab.append(str(c))
        c += 1
    linhas = []
    nomes = ["treino", "validação", "teste"]
    conts = [cont_tr, cont_val, cont_te]
    i = 0
    while i < 3:
        linha = [nomes[i]]
        for v in conts[i]:
            linha.append(numero_br(v))
        linhas.append(linha)
        i += 1
    salvar_tabela("dados_classes", cab, linhas, "Número de imagens por classe em cada conjunto.", "tab:classes")

    grafico_exemplos(d["X_tr_bruto"], d["y_tr"])
    grafico_distribuicao(cont_tr, cont_val, cont_te)

    info = {}
    info["n_treino"] = len(d["y_tr"])
    info["n_validacao"] = len(d["y_val"])
    info["n_teste"] = len(d["y_te"])
    info["media"] = d["media"]
    info["desvio"] = d["desvio"]
    info["por_classe"] = {"treino": cont_tr, "validacao": cont_val, "teste": cont_te}
    return info

def busca_mlp(d, dispositivo):
    print("\n=== Busca de arquiteturas do MLP")
    loader_tr = criar_loader(d["X_tr"], d["y_tr"], LOTE_MLP, True)
    loader_val = criar_loader(d["X_val"], d["y_val"], 1000, False)

    busca = {}
    melhor = None
    for ocultas in ARQUITETURAS_MLP:
        nome = nome_arquitetura(ocultas)
        print(f"MLP {nome}")
        fixar_semente()
        modelo = MLP(ocultas)
        iniciar_pesos(modelo)
        hist = treinar(modelo, loader_tr, loader_val, EPOCAS_BUSCA, LR_MLP, MOMENTO_MLP, dispositivo, mostrar=False)
        ep = hist["melhor_epoca"] - 1
        r = {}
        r["ocultas"] = ocultas
        r["parametros"] = contar_parametros(modelo)
        r["acc_val"] = hist["acc_val"][ep]
        r["perda_val"] = hist["perda_val"][ep]
        r["melhor_epoca"] = hist["melhor_epoca"]
        r["tempo"] = hist["tempo_total"]
        r["hist"] = hist
        busca[nome] = r
        print(f"  parâmetros={r['parametros']}  acc val={r['acc_val']:.4f}  perda val={r['perda_val']:.4f}  época={r['melhor_epoca']}  tempo={r['tempo']:.1f} s")

        if melhor is None:
            melhor = nome
        elif r["acc_val"] > busca[melhor]["acc_val"]:
            melhor = nome
        elif r["acc_val"] == busca[melhor]["acc_val"] and r["parametros"] < busca[melhor]["parametros"]:
            melhor = nome

    linhas = []
    for nome in busca:
        r = busca[nome]
        marca = ""
        if nome == melhor:
            marca = " *"
        linhas.append([nome.replace("x", " $\\times$ ") + marca, numero_br(r["parametros"]), numero_br(r["acc_val"], 4),
                       numero_br(r["perda_val"], 4), str(r["melhor_epoca"]), numero_br(r["tempo"], 1)])
    salvar_tabela("mlp_busca", ["camadas ocultas", "parâmetros", "acurácia val.", "perda val.", "época", "tempo (s)"], linhas,
                  "Busca de arquiteturas do MLP (" + str(EPOCAS_BUSCA) + " épocas, SGD, $\\eta$ = " + numero_br(LR_MLP, 2) +
                  ", lote " + str(LOTE_MLP) + "). Valores na época de menor perda de validação; * = escolhida.", "tab:busca")
    grafico_busca(busca)
    print(f"escolhida: {melhor}")
    return busca, busca[melhor]["ocultas"]

def rodar_modelo(tag, titulo, modelo, d, lote, epocas, lr, momento, dispositivo):
    print(f"\n=== {titulo}")
    loader_tr = criar_loader(d["X_tr"], d["y_tr"], lote, True)
    loader_val = criar_loader(d["X_val"], d["y_val"], 1000, False)
    loader_te = criar_loader(d["X_te"], d["y_te"], 1000, False)

    print(f"parâmetros: {contar_parametros(modelo)}")
    hist = treinar(modelo, loader_tr, loader_val, epocas, lr, momento, dispositivo)
    print(f"menor perda de validação na época {hist['melhor_epoca']}; tempo total {hist['tempo_total']:.1f} s")

    perda_te, y_real, y_prev = avaliar(modelo, loader_te, dispositivo)
    m = calcular_metricas(y_real, y_prev)
    m["perda"] = perda_te
    print(f"teste: acc={m['acuracia']:.4f}  prec={m['precisao']:.4f}  rev={m['revocacao']:.4f}  f1={m['f1']:.4f}  erros={m['erros']}")

    pares = pares_mais_confundidos(m["confusao"], 5)
    texto = ""
    for p in pares:
        texto = texto + f"{p[0]}->{p[1]} ({p[2]})  "
    print("pares mais confundidos: " + texto)

    grafico_perda(hist, titulo, tag + "_perda")
    grafico_confusao(m["confusao"], titulo, tag + "_confusao")
    exemplos = []
    for p in pares:
        exemplos.append([p[0], p[1], indices_erros(y_real, y_prev, p[0], p[1])])
    grafico_erros(d["X_te_bruto"], exemplos, tag + "_erros")

    r = {}
    r["parametros"] = contar_parametros(modelo)
    r["hist"] = hist
    r["teste"] = m
    r["pares"] = pares
    r["y_prev"] = y_prev
    return r

def tabelas_finais(res, d):
    linhas = []
    for tag in ["mlp", "cnn"]:
        r = res[tag]
        t = r["teste"]
        linhas.append([r["nome"], numero_br(r["parametros"]), numero_br(t["acuracia"], 4), numero_br(t["precisao"], 4),
                       numero_br(t["revocacao"], 4), numero_br(t["f1"], 4), numero_br(t["perda"], 4)])
    salvar_tabela("metricas_teste", ["modelo", "parâmetros", "acurácia", "precisão", "revocação", "F1", "perda"], linhas,
                  "Métricas no conjunto de teste (precisão, revocação e F1 macro).", "tab:metricas")

    mlp = res["mlp"]
    cnn = res["cnn"]
    linhas = []
    linhas.append(["parâmetros", numero_br(mlp["parametros"]), numero_br(cnn["parametros"])])
    linhas.append(["épocas", str(EPOCAS_MLP), str(EPOCAS_CNN)])
    linhas.append(["época escolhida", str(mlp["hist"]["melhor_epoca"]), str(cnn["hist"]["melhor_epoca"])])
    linhas.append(["tempo por época (s)", numero_br(mlp["hist"]["tempo_epoca"], 2), numero_br(cnn["hist"]["tempo_epoca"], 2)])
    linhas.append(["tempo total (s)", numero_br(mlp["hist"]["tempo_total"], 1), numero_br(cnn["hist"]["tempo_total"], 1)])
    linhas.append(["acurácia", numero_br(mlp["teste"]["acuracia"], 4), numero_br(cnn["teste"]["acuracia"], 4)])
    linhas.append(["F1 macro", numero_br(mlp["teste"]["f1"], 4), numero_br(cnn["teste"]["f1"], 4)])
    linhas.append(["erros (de 10.000)", numero_br(mlp["teste"]["erros"]), numero_br(cnn["teste"]["erros"])])
    salvar_tabela("comparacao", ["", mlp["nome"], cnn["nome"]], linhas,
                  "Comparação entre o MLP e a LeNet no conjunto de teste.", "tab:comparacao")

    linhas = []
    i = 0
    while i < 5:
        linha = []
        for tag in ["mlp", "cnn"]:
            if i < len(res[tag]["pares"]):
                p = res[tag]["pares"][i]
                linha.append(str(p[0]) + " $\\to$ " + str(p[1]))
                linha.append(str(p[2]))
            else:
                linha.append("--")
                linha.append("--")
        linhas.append(linha)
        i += 1
    salvar_tabela("pares_confundidos", ["MLP (real $\\to$ previsto)", "qtd.", "LeNet (real $\\to$ previsto)", "qtd."], linhas,
                  "Pares de dígitos mais confundidos no teste.", "tab:pares", "lclc")

    linhas = []
    c = 0
    while c < N_CLASSES:
        linhas.append([str(c), numero_br(mlp["teste"]["f1_classe"][c], 4), numero_br(cnn["teste"]["f1_classe"][c], 4)])
        c += 1
    salvar_tabela("f1_classes", ["dígito", "F1 MLP", "F1 LeNet"], linhas, "F1 de cada classe no teste.", "tab:f1classes")

    linhas = []
    for l in res["camadas_cnn"]:
        linhas.append([l[0], l[1], numero_br(l[2])])
    salvar_tabela("lenet_parametros", ["posição", "camada", "parâmetros"], linhas, "Parâmetros de cada camada da LeNet.", "tab:lenet")

if __name__ == "__main__":
    if not os.path.exists("resultados"):
        os.makedirs("resultados")
    dispositivo = escolher_dispositivo()
    print(f"dispositivo: {dispositivo}")

    d = preparar_dados()
    saida = {}
    saida["dispositivo"] = dispositivo
    saida["dados"] = descrever_dados(d)

    if RODAR_BUSCA_MLP:
        busca, ocultas = busca_mlp(d, dispositivo)
        saida["busca_mlp"] = {}
        for nome in busca:
            r = dict(busca[nome])
            del r["hist"]
            saida["busca_mlp"][nome] = r
    else:
        ocultas = ARQ_MLP_FIXA

    res = {}
    fixar_semente()
    mlp = MLP(ocultas)
    iniciar_pesos(mlp)
    nome_mlp = "MLP " + nome_arquitetura(ocultas).replace("x", " $\\times$ ")
    res["mlp"] = rodar_modelo("mlp", "MLP " + nome_arquitetura(ocultas), mlp, d, LOTE_MLP, EPOCAS_MLP, LR_MLP, MOMENTO_MLP, dispositivo)
    res["mlp"]["nome"] = nome_mlp

    fixar_semente()
    cnn = LeNet()
    iniciar_pesos(cnn)
    res["cnn"] = rodar_modelo("cnn", "LeNet", cnn, d, LOTE_CNN, EPOCAS_CNN, LR_CNN, MOMENTO_CNN, dispositivo)
    res["cnn"]["nome"] = "LeNet"
    res["camadas_cnn"] = parametros_por_camada(cnn)

    tabelas_finais(res, d)
    grafico_comparacao(res)

    for tag in ["mlp", "cnn"]:
        r = dict(res[tag])
        del r["y_prev"]
        saida[tag] = r
    saida["camadas_cnn"] = res["camadas_cnn"]

    arq = open("resultados/resultados.json", "w", encoding="utf-8")
    json.dump(saida, arq, indent=2, ensure_ascii=False, default=str)
    arq.close()
    print("\nfiguras e tabelas salvas na pasta resultados/")
