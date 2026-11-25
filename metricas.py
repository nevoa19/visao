from dados import N_CLASSES

def matriz_confusao(y_real, y_prev):
    cm = []
    a = 0
    while a < N_CLASSES:
        linha = []
        b = 0
        while b < N_CLASSES:
            linha.append(0)
            b += 1
        cm.append(linha)
        a += 1
    i = 0
    while i < len(y_real):
        cm[y_real[i]][y_prev[i]] += 1
        i += 1
    return cm

def calcular_metricas(y_real, y_prev):
    cm = matriz_confusao(y_real, y_prev)
    total = len(y_real)
    acertos = 0
    precisoes = []
    revocacoes = []
    f1s = []
    k = 0
    while k < N_CLASSES:
        vp = cm[k][k]
        acertos += vp
        soma_coluna = 0 
        soma_linha = 0   
        j = 0
        while j < N_CLASSES:
            soma_coluna += cm[j][k]
            soma_linha += cm[k][j]
            j += 1
        p = 0.0
        if soma_coluna > 0:
            p = vp / soma_coluna
        r = 0.0
        if soma_linha > 0:
            r = vp / soma_linha
        f = 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
        precisoes.append(p)
        revocacoes.append(r)
        f1s.append(f)
        k += 1

    soma_p = 0
    soma_r = 0
    soma_f = 0
    k = 0
    while k < N_CLASSES:
        soma_p = soma_p + precisoes[k]
        soma_r = soma_r + revocacoes[k]
        soma_f = soma_f + f1s[k]
        k += 1

    m = {}
    m["acuracia"] = acertos / total
    m["precisao"] = soma_p / N_CLASSES
    m["revocacao"] = soma_r / N_CLASSES
    m["f1"] = soma_f / N_CLASSES
    m["erros"] = total - acertos
    m["precisao_classe"] = precisoes
    m["revocacao_classe"] = revocacoes
    m["f1_classe"] = f1s
    m["confusao"] = cm
    return m

def pares_mais_confundidos(cm, quantos=5):
    pares = []
    a = 0
    while a < N_CLASSES:
        b = 0
        while b < N_CLASSES:
            if a != b and cm[a][b] > 0:
                pares.append([a, b, cm[a][b]])
            b += 1
        a += 1

    escolhidos = []
    while len(escolhidos) < quantos and len(pares) > 0:
        maior = 0
        i = 1
        while i < len(pares):
            if pares[i][2] > pares[maior][2]:
                maior = i
            i += 1
        escolhidos.append(pares[maior])
        pares.pop(maior)
    return escolhidos

def indices_erros(y_real, y_prev, real, previsto):
    idx = []
    i = 0
    while i < len(y_real):
        if y_real[i] == real and y_prev[i] == previsto:
            idx.append(i)
        i += 1
    return idx
