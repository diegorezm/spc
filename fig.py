"""
Modulo para criação de graficos/figuras para analise de espectroscopia de absorção
utilizando PLSR.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.cross_decomposition import PLSRegression

from read import SpectroscopyData


def loading_fig(data: SpectroscopyData, ncomp: int, title: str = "Loading Plot", xlabel: str = "Número de onda", ylabel: str = "Loading Values"):
    """
    Retorna um gráfico de carregamentos de previsão (loading) do PLS.

    Parameters:
      data: Dados espectroscópicos.
      ncomp (int): Número de componentes principais a serem utilizados no PLS.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.

    Returns:
      plt.Figure: O objeto da figura do gráfico.
    """
    g = data.group_ids
    r = data.abss
    wn = data.wn

    pls = PLSRegression(n_components=ncomp)
    pls.fit(r, g)
    loading = pls.x_loadings_

    fig, ax = plt.subplots()
    offset = 0
    leng = []
    for i in range(loading.shape[1]):
        # Plotando os carregamentos
        ax.plot(wn, offset + loading[:, i])
        # Calculando o offset para os carregamentos
        offset += np.abs(loading[:, i].min()) + np.abs(loading[:, i].max())
        # Adicionando o rótulo do predictor
        leng.append(f'predictor loading_{i + 1}')

    ax.legend(leng)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def scatter_fig(data: SpectroscopyData, a: int, b: int, ncomp: int, title: str = "PLSR Scatter Plot", xlabel: str = "X Score", ylabel: str = "Y Score"):
    """
    Retorna um gráfico de dispersão das pontuações PLS.

    Parameters:
      data (SpectroscopyData): Dados espectroscópicos.
      a (int): Índice da pontuação do eixo x.
      b (int): Índice da pontuação do eixo y.
      ncomp (int): Número de componentes principais a serem utilizados no PLS.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.

    Returns:
      plt.Figure: O objeto da figura do gráfico.
    """

    g = data.group_ids
    r = data.abss

    pls = PLSRegression(n_components=ncomp)
    pls.fit(r, g)
    scatter_scores = pls.x_scores_

    fig, ax = plt.subplots()
    unique_groups = np.unique(g)
    for i in unique_groups:
        # Selecionando os dados para o grupo atual
        sel = g == i
        # Plotando os pontos no gráfico de dispersão
        ax.scatter(scatter_scores[sel, a - 1],
                   scatter_scores[sel, b - 1], color=data.colors[sel][0])

    ax.legend([f'Group {i}' for i in unique_groups])
    ax.set_xlabel(f'{a} {xlabel}', fontsize=12)
    ax.set_ylabel(f'{b} {ylabel}', fontsize=12)
    ax.set_title(title)

    return fig


def fit_fig(data: SpectroscopyData, ncomp: int, title: str = "PLSR Fit Plot", xlabel: str = "Y Class", ylabel: str = "Y Predicted"):
    """
    Retorna um gráfico de ajuste do PLS com histograma.

    Parameters:
      data (SpectroscopyData): Dados espectroscópicos.
      ncomp (int): Número de componentes principais a serem utilizados no PLS.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.

    Returns:
      plt.Figure: O objeto da figura do gráfico.
    """

    g = data.group_ids
    r = data.abss

    pls = PLSRegression(n_components=ncomp)
    pls.fit(r, g)

    Y_pred = pls.predict(r)

    # Criando o gráfico de dispersão e o histograma, respectivamente
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    unique_groups = np.unique(g)

    for i in unique_groups:
        sel = g == i
        ax1.scatter(g[sel], Y_pred[sel], color=data.colors[sel][0])

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_xticks(unique_groups)
    ax1.set_xticklabels([f'Group {i}' for i in unique_groups])

    # Histograma
    for i in unique_groups:
        sel = g == i
        ax2.hist(Y_pred[sel], alpha=0.5)

    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Histogram', fontsize=12)
    ax2.set_xticks(unique_groups)
    ax2.set_xticklabels([f'Group {i}' for i in unique_groups])

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def coeff_fig(data: SpectroscopyData, ncomp: int, title: str = "PLSR Coefficients Plot", xlabel: str = "Número de onda", ylabel: str = "Coefficient Value"):
    """
    Retorna um gráfico de coeficientes do PLS.

    Parameters:
      data (SpectroscopyData): Dados espectroscópicos.
      ncomp (int): Número de componentes principais a serem utilizados no PLS.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.

    Returns:
      plt.Figure: O objeto da figura do gráfico.
    """

    g = data.group_ids
    r = data.abss
    wn = data.wn

    pls = PLSRegression(n_components=ncomp)
    pls.fit(r, g)

    fig, ax = plt.subplots()

    # Plotando os coeficientes e a linha zero
    zeroline = np.zeros_like(pls.coef_)
    ax.plot(wn, pls.coef_, wn, zeroline)
    ax.set_xlim((wn[0], wn[-1]))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)

    return fig


def loading_fig_with_peaks(data: SpectroscopyData, ncomp: int, peaks: dict[tuple[int, int], float],
                           title: str = "Loading Plot with Peaks", xlabel: str = "Número de onda", ylabel: str = "Loading Values"):
    """
    Retorna um gráfico de carregamentos de previsão (loading) do PLS, destacando os picos.

    Parameters:
      data: Dados espectroscópicos.
      ncomp (int): Número de componentes principais a serem utilizados no PLS.
      peaks (dict): Dicionário com as coordenadas (i, j) dos picos.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.

    Returns:
      plt.Figure: O objeto da figura do gráfico.
    """
    g = data.group_ids
    r = data.abss
    wn = data.wn

    pls = PLSRegression(n_components=ncomp)
    pls.fit(r, g)
    loading = pls.x_loadings_

    fig, ax = plt.subplots()
    offset = 0
    leng = []

    mean = np.mean(loading)
    std = np.std(loading)
    height = mean - std

    for i in range(loading.shape[1]):
        # Plotando os carregamentos
        ax.plot(wn, offset + loading[:, i], label=f'Predictor Loading {i + 1}')

        # Identificando picos no carregamento
        # You can adjust the height threshold if needed
        p, _ = find_peaks(loading[:, i], height=height)

        # Plotando os picos
        ax.plot(wn[p], offset + loading[p, i],
                'ro')  # Red circles for peaks

        # Calculando o offset para os carregamentos
        offset += np.abs(loading[:, i].min()) + np.abs(loading[:, i].max())

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    return fig
