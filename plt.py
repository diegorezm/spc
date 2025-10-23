"""
Modulo para o plot de figuras para analise de espectroscopia utilizando PLSR
"""

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_validate
from read import SpectroscopyData
import fig
import numpy as np


def loading_plot(
    data: SpectroscopyData,
    ncomp: int,
    title: str = "Loading Plot",
    xlabel: str = "Número de onda",
    ylabel: str = "Loading Values",
):
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


def scatter_plot(
    data: SpectroscopyData,
    a: int,
    b: int,
    ncomp: int,
    title: str = "PLSR Scatter Plot",
    xlabel: str = "X Score",
    ylabel: str = "Y Score",
):
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


def fit_plot(
    data: SpectroscopyData,
    ncomp: int,
    title: str = "PLSR Fit Plot",
    xlabel: str = "Y Class",
    ylabel: str = "Y Predicted",
):
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


def coeff_plot(
    data: SpectroscopyData,
    ncomp: int,
    title: str = "PLSR Coefficients Plot",
    xlabel: str = "Número de onda",
    ylabel: str = "Valor do coeficiente",
):
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
    rsme = []

    for i in range(1, ncomp + 1):
        pls = PLSRegression(n_components=i)
        results = cross_validate(
            pls,
            r,
            g,
            cv=kfold,
            return_train_score=True,
            return_estimator=True,
            scoring="neg_mean_squared_error",
        )
        rsme.append(results)
    return rsme


def loading_plot_with_peaks(
    data: SpectroscopyData,
    ncomp: int,
    peaks: dict[tuple[int, int], float],
    title: str = "PLSR Loading Plot",
    xlabel: str = "X Score",
    ylabel: str = "Loading Values",
):
    f = fig.loading_fig_with_peaks(data, ncomp, peaks, title, xlabel, ylabel)
    plt.show()
    return f


def plot_peaks(
    data: SpectroscopyData,
    peaks: dict,
    title: str = "Detected Peaks",
    xlabel: str = "Número de onda",
    ylabel: str = "Absorbância",
):
    """
    Plota os dados de espectroscopia com os picos detectados.

    Parameters:
        data (SpectroscopyData): Dados espectroscópicos.
        peaks (dict): Dicionário com as coordenadas (i, j) dos picos detectados.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo x.
        ylabel (str): Rótulo do eixo y.
    """
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Plotando cada espectro
    for i, spectrum in enumerate(data.abss):
        ax.plot(data.wn, spectrum, label=f"Grupo {data.group_ids[i]}", alpha=0.5)

    # Plotando os picos detectados
    for (i, j), value in peaks.items():
        # Add a red dot for each peak
        # Only label the first peak for clarity
        ax.plot(
            data.wn[j],
            value,
            "ro",
            markersize=8,
            label=f"Pico {j} em Grupo {i}" if j == 0 else "",
        )

    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="upper right", fontsize="small", frameon=True
    )  # Legend on the upper right
    plt.tight_layout()
    plt.grid()  # Optionally, add a grid for better visibility
    plt.show()


def std_plot(data: SpectroscopyData) -> None:
    """
    Calcula o desvio padrão dos dados de espectroscopia e plota os resultados.

    Parametros:
        data (SpectroscopyData): O objeto contendo os espectros de absorbância e números de onda.
    """
    # Obtendo os grupos únicos de IDs
    group_ids = np.unique(data.group_ids)

    plt.figure(figsize=(12, 6))

    # Para cada grupo, calcula média e desvio padrão
    for group_id in group_ids:
        # Seleciona os dados para o grupo atual
        group_data = data.abss[data.group_ids == group_id]

        # Calcula a média e o desvio padrão
        mean_values = np.mean(group_data, axis=0)
        std_values = np.std(group_data, axis=0)

        # Plota a média
        plt.plot(data.wn, mean_values, label=f"Média do Grupo {int(group_id)}")

        # Plota a sombra do desvio padrão
        plt.fill_between(
            data.wn, mean_values - std_values, mean_values + std_values, alpha=0.2
        )

    # Configurações do gráfico
    plt.title("Desvio Padrão dos Espectros de Absorbância")
    plt.xlabel("Número de Onda (cm⁻¹)")
    plt.ylabel("Absorbância")
    plt.axhline(
        0, color="black", linewidth=0.5, linestyle="--"
    )  # Linha horizontal em y=0

    plt.legend()
    plt.grid()
    plt.show()
