"""
Modulo para analise de espectroscopia de absorção utilizando PCA.
"""
import numpy as np
from read import SpectroscopyData
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

def scores_fig(dados: SpectroscopyData, 
               a: int, 
               b: int, 
               title: str = "PCA Scores Plot"):
    """
    Retorna a figura das pontuações da PCA.

    Parameters:
        dados (SpectroscopyData): Dados espectroscópicos.
        a (int): Índice do eixo x.
        b (int): Índice do eixo y.
        title (str): Título do gráfico.
    """
    pcadata = pca(dados.abss)
    latent = np.round(pcadata[2] / pcadata[2].sum(), 5)
    leng = str(dados.args).split('::')
    unique_groups = np.unique(dados.group_ids)
    
    # Criando a figura e o eixo
    fig, ax = plt.subplots()
    
    for i in unique_groups:
        sel = dados.group_ids == i
        ax.scatter(pcadata[0][sel, a - 1], pcadata[0][sel, b - 1], color=dados.colors[sel][0], label=f'Group {i}')
    
    ax.legend(leng)
    ax.set_xlabel(f'PC {a} ({100 * latent[0, a - 1]:.2f}%)')
    ax.set_ylabel(f'PC {b} ({100 * latent[0, b - 1]:.2f}%)')
    ax.set_title(title)
    
    return fig

def scores_plot(dados: SpectroscopyData, a: int, b: int, title: str = "PCA Scores Plot"):
    """
    Plota as pontuações da PCA.

    Parameters:
        dados (SpectroscopyData): Dados espectroscópicos.
        a (int): Índice do eixo x.
        b (int): Índice do eixo y.
        title (str): Título do gráfico.
    """
    fig = scores_fig(dados, a, b, title)
    plt.show()
    return fig

def loading_fig(data: SpectroscopyData, 
                ncomp: list[int], 
                title: str = "Loading Plot", 
                xlabel: str = "Número de onda (cm^{-1})", 
                ylabel: str = "Loading Values"):
    """
    Retorna um gráfico de carregamentos de previsão (loading) do PCA.

    Parameters:
      data: Dados espectroscópicos.
      ncomp (int): Número de componentes principais a serem utilizados no PCA.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.

    Returns:
      plt.Figure: O objeto da figura do gráfico.
    """
    # Extraindo dados
    wn = data.wn
    
    # Executando PCA
    pcadata = pca(data.abss)
    coeff = pcadata[1]

    # Criando o gráfico
    lengs = []
    for i in ncomp:
        # Plotando os carregamentos
        plt.plot(wn, coeff[:, i])
        lengs.append(f'Loading pc {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend(lengs)
    plt.ylabel(ylabel)
    
    return plt.gcf()

def loading_plt(data: SpectroscopyData,ncomp: list[int],  title: str = "Loading Plot", 
                xlabel: str = "Número de onda (cm^{-1})", 
                ylabel: str = "Loading Values"):
    fig = loading_fig(data, ncomp, title, xlabel, ylabel)
    plt.show()
    return fig

__all__ = ['pca', 'scores_plot', 'scores_fig', 'loading_fig', 'loading_plt']
