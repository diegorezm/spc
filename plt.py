"""
Modulo para o plot de figuras para analise de espectroscopia
"""

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate

import fig
from read import SpectroscopyData


def loading_plot(data: SpectroscopyData, ncomp: int, 
                 title: str = "Loading Plot", 
                 xlabel: str = "Número de onda", 
                 ylabel: str = "Loading Values"):
    """
     Plota os carregamentos de previsão (loading) do PLS.

    Parameters:
      data: Dados espectroscópicos.
      ncomp (int): Número de componentes principais a serem utilizados no PLS.
      title (str): Título do gráfico.
      xlabel (str): Rótulo do eixo x.
      ylabel (str): Rótulo do eixo y.
    """
    f = fig.loading_fig(data, ncomp, title, xlabel, ylabel)
    plt.show()
    return f

def scatter_plot(data: SpectroscopyData, a: int, b: int, ncomp: int, title: str = "PLSR Scatter Plot", xlabel: str = "X Score", ylabel: str = "Y Score"):
    """
    Plota um gráfico de dispersão das pontuações PLS.

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

    f = fig.scatter_fig(data, a, b, ncomp, title, xlabel, ylabel)
    plt.show()
    return f

def fit_plot(data: SpectroscopyData, ncomp: int, title: str = "PLSR Fit Plot", xlabel: str = "Y Class", ylabel: str = "Y Predicted"):
    """
    Plota o gráfico de ajuste e o histograma das previsões do PLS.

    Parameters:
    data (SpectroscopyData): Dados espectroscópicos.
    ncomp (int): Número de componentes principais a serem utilizados no PLS.
    title (str): Título do gráfico.
    xlabel (str): Rótulo do eixo x.
    ylabel (str): Rótulo do eixo y.
    """
    f = fig.fit_fig(data, ncomp, title, xlabel, ylabel)
    plt.show()
    return f

def coeff_plot(data: SpectroscopyData, ncomp: int, title: str = "PLSR Coefficients Plot", xlabel: str = "Número de onda", ylabel: str = "Valor do coeficiente"):
    """
    Plota os coeficientes da regressão do PLS.

    Parameters:
    data (SpectroscopyData): Dados espectroscópicos.
    ncomp (int): Número de componentes principais a serem utilizados no PLS.
    title (str): Título do gráfico.
    xlabel (str): Rótulo do eixo x.
    ylabel (str): Rótulo do eixo y.
    """
    f = fig.coeff_fig(data, ncomp, title, xlabel, ylabel)
    plt.show()
    return f

def cross_validation(data: SpectroscopyData, ncomp: int, kfold: int):
    """
    Realiza validação cruzada para o modelo PLS.

    Parameters:
    data (SpectroscopyData): Dados espectroscópicos.
    ncomp (int): Número máximo de componentes principais a serem utilizados.
    kfold (int): Número de divisões para a validação cruzada.

    Returns:
    List: Lista de resultados de validação cruzada.
    """
    g = data.group_ids
    r = data.abss
    rsme = ['none']
    
    for i in range(1, ncomp + 1):
        pls = PLSRegression(n_components=i)
        rsme.append(cross_validate(pls, r, g, cv=kfold, return_train_score=True, return_estimator=True, scoring='neg_mean_squared_error'))
    
    return rsme
