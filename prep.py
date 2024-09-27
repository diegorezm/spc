"""
Modulo de preparação de dados de espectroscopia.
"""

import numpy as np
from numpy.matlib import repmat
from scipy.signal import savgol_coeffs
from scipy.sparse import spdiags

from read import SpectroscopyData


def group(d1: SpectroscopyData, *other_data: SpectroscopyData) -> SpectroscopyData:
    """
    Concatena os dados de espectroscopia de múltiplos grupos em um único objeto `SpectroscopyData`.

    O número de onda (`wn`) do primeiro grupo fornecido será usado para todos os grupos subsequentes.
    Esta função agrega os valores de absorbância de cada grupo e gera IDs de grupo únicos para cada
    conjunto de dados. Além disso, concatena os nomes dos arquivos associados a cada grupo para
    rastreamento.
    """
    # Copiar os dados do primeiro grupo
    abss = d1.abss.copy()
    group_ids = d1.group_ids.copy()
    args = d1.args.copy()
    wn = d1.wn
    colors = d1.colors.copy()

    for i,data in enumerate(other_data, start=2):
        # Concatenar os valores de absorbância, IDs dos grupos e os nomes dos arquivos
        abss = np.concatenate((abss, data.abss))
        group_ids = np.concatenate((group_ids, i * np.ones(len(data.abss))))
        # Grupos são identificados por um número, "::" é utilizado para separar os grupos
        args = np.char.add(args,"::")
        args = np.char.add(args, data.args)
        colors = np.concatenate((colors, data.colors))
    # Vai semmpre utilizar o número de onda do primeiro grupo
    return SpectroscopyData(abss, wn, group_ids, args, colors)

def cut(data: SpectroscopyData, a: float, b: float) -> SpectroscopyData:
    """
    Faz a restrição espectral dos dados de espectroscopia.
    """
    sel = (data.wn > a) & (data.wn < b)
    return SpectroscopyData(data.abss[:, sel], data.wn[sel], data.group_ids, data.args, data.colors)

def golay(data: SpectroscopyData, diff: int, order: int, win: int) -> SpectroscopyData:
    """
    Aplica o filtro de Savitzky-Golay aos dados de espectroscopia.
    
    Parametros:
        diff: int, derivada
        order: int, ordem
        win: int, janela
    """
    n = int((win - 1) / 2)
    # Coeficientes de Savitzky-Golay para a derivada especificada
    sgcoeff = savgol_coeffs(win, order, deriv=diff)[:, None]
    
    # Replicar os coeficientes para todas as colunas de absorbância
    sgcoeff = repmat(sgcoeff, 1, data.abss.shape[1])
    
    # Criar a matriz esparsa diagonal com os coeficientes
    diags = np.arange(-n, n + 1)
    d = spdiags(sgcoeff, diags, data.abss.shape[1], data.abss.shape[1]).toarray()
    
    # Zero padding nas bordas para evitar problemas de contorno
    d[:, 0:n] = 0
    d[:, data.abss.shape[1] - 5:data.abss.shape[1]] = 0
    
    # Aplicar o filtro aos dados de absorbância
    data.abss = np.dot(data.abss, d)
    
    return data

def norm2r(data: SpectroscopyData, a: float, b: float, c: float, d: float) -> SpectroscopyData:
    """
    Normaliza os dados de espectroscopia em duas regiões.

    Parametros:
        data: Dados de espectroscopia
        a, b: float, limites da primeira região
        c, d: float, limites da segunda região
    """
    # Seleciona os dados dentro da primeira região
    sel = (data.wn > a) & (data.wn < b)

    # Normaliza os dados dentro da primeira região
    r1 = data.abss[sel]
    wn1 = data.wn[sel][:, None]
    media = np.mean(r1, axis=1)
    std = np.std(r1, axis=1)
    r1 = np.divide((r1 - media[:, None]), std[:, None])
    
    # Seleciona os dados dentro da segunda região
    sel = (data.wn > c) & (data.wn < d)

    # Normaliza os dados dentro da segunda região
    r2 = data.abss[sel]
    wn2 = data.wn[sel][:, None]
    media = np.mean(r2, axis=1)
    std = np.std(r2, axis=1)
    r2 = np.divide((r2 - media[:, None]), std[:, None])
    
    # Concatena os dados normalizados
    data.abss = np.column_stack((r1, r2))
    data.wn = np.vstack((wn1, wn2))
    data.wn = data.wn.reshape(-1)
    
    return data

def norm_vec(data: SpectroscopyData) -> SpectroscopyData:
    """
    Normaliza os dados de espectroscopia vetorialmente.
    """
    # Obter o array de absorbância
    r = data.abss

    # Calcular a norma de cada vetor
    norma = np.sqrt((r * r).sum(axis=1)).reshape(-1, 1)

    # Replicar a norma ao longo das colunas de r
    rnorm = np.tile(norma, (1, r.shape[1]))

    # Normalizar os dados de absorbância
    data.abss = r / rnorm
    return data

def snv(data: SpectroscopyData) -> SpectroscopyData:
    """
    Aplica a normalização Standard Normal Variate (SNV) aos espectros de absorbância.
    A SNV remove o efeito de dispersão multiplicativa dos espectros, normalizando cada espectro individualmente.
    """
    spc = data.abss
    media = np.mean(spc, axis=1)
    std = np.std(spc, axis=1)
    data.abss = np.divide((spc - media[:, None]), std[:, None])
    return data

def offset(data: SpectroscopyData, ini: float, fim: float) -> SpectroscopyData:
    """
    Remove o offset das regiões entre os números de onda especificados.

    A função subtrai o valor mínimo do espectro na região entre os números de onda 'ini' e 'fim' 
    para cada espectro de absorbância.

    Parametros:
    data (SpectroscopyData): O objeto contendo os espectros de absorbância e números de onda.
        ini (float): O limite inferior do intervalo (em número de onda).
        fim (float): O limite superior do intervalo (em número de onda).
    """
    # Seleciona a região entre ini e fim
    sel = np.logical_and(data.wn > ini, data.wn < fim)
    
    # Extrai os espectros da região selecionada
    r = data.abss[:, sel]
    
    # Calcula o valor mínimo para cada espectro na região selecionada
    minimo = np.min(r, axis=1).reshape(-1, 1)
    
    # Cria um array com o valor mínimo replicado para subtrair de todos os pontos do espectro
    minimo_tiled = np.tile(minimo, (1, data.abss.shape[1]))

    # Subtrai o valor mínimo de cada espectro original
    data.abss = data.abss - minimo_tiled
    
    return data

def dsample(data: SpectroscopyData, k: int) -> SpectroscopyData:
    """
    Faz a subamostragem dos dados de espectroscopia.
    Reduz os dados de absorbância e os números de onda com base em um fator k.
    Parameters:
        data: Dados de espectroscopia
        k: Fator de subamostragem
    """
    data.abss = data.abss[:, ::k]
    data.wn = data.wn[::k]
    return data

__all__ = ["group", "cut", "golay", "norm2r", "norm_vec", "snv", "offset", "dsample"]
