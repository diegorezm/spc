"""
Modulo para analise de espectroscopia de absorção utilizando PCA.
"""
import numpy as np
from age import SpectroscopyData
import matplotlib.pyplot as plt

def pca(abss: np.ndarray):
    """
    Realiza a análise de componentes principais (PCA) nos dados espectroscópicos.

    Parameters:
        abss (np.ndarray): Matriz de dados espectroscópicos.

    Returns:
        tuple: Contendo as pontuações, os coeficientes e as raízes latentes da PCA.
    """
    rmean = abss - abss.mean(0)
    rcov = np.cov(rmean.T)
    _,latent,coeff = np.linalg.svd(rcov)
    latent = latent.reshape(1,-1)
    coeff = coeff.T*np.sqrt(latent)
    scores = abss @ (coeff)
    return scores, coeff, np.sqrt(latent)

def scores_fig(dados: SpectroscopyData, a: int, b: int, title: str = "PCA Scores Plot"):
    """
    Retorna a figura das pontuações da PCA.

    Parameters:
        dados (SpectroscopyData): Dados espectroscópicos.
        a (int): Índice do eixo x.
        b (int): Índice do eixo y.
        title (str): Título do gráfico.
    """
    pcadata = pca(dados.abss)
    latent = np.round(pcadata[2]/pcadata[2].sum(),5)
    leng = str(dados.args)
    leng = leng.split('::')
    unique_groups = np.unique(dados.group_ids)
    for i in unique_groups:
        sel = dados.group_ids == i
        plt.scatter(pcadata[0][sel,a-1],pcadata[0][sel,b-1],color=dados.colors[sel][0])
    plt.legend(leng)
    plt.xlabel('pc_' +str(a) + '  ' + str(100*latent[0,a-1])[:5] + '%')
    plt.ylabel('pc_' +str(b)+ '  ' + str(100*latent[0,b-1])[:5] + '%')
    plt.title(title)
    plt.show()
    return plt.gcf()

def scores_plot(dados: SpectroscopyData, a: int, b: int, title: str = "PCA Scores Plot"):
    """
    Plota as pontuações da PCA.

    Parameters:
        dados (SpectroscopyData): Dados espectroscópicos.
        a (int): Índice do eixo x.
        b (int): Índice do eixo y.
        title (str): Título do gráfico.
    """
    scores_fig(dados, a, b, title)

__all__ = ['pca', 'scores_plot', 'scores_fig']