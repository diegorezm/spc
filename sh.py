import matplotlib.pyplot as plt
from numpy import unique, arange
from scipy.signal import find_peaks_cwt

from read import SpectroscopyData


def mplot(data: SpectroscopyData):
    """
    Função para plotar o espectro medio
    """
    unique_groups = unique(data.group_ids)
    for i in unique_groups:
        sel = data.group_ids == i
        plt.plot(data.wn, data.abss[sel, :].mean(0))
    legenda = str(data.args)
    legenda = legenda.split('::')
    plt.legend(legenda)
    plt.xlabel('numero de onda (cm^{-1})')
    plt.show()


def mplot_peaks_fig(data: SpectroscopyData):
    """
    Gera o gráfico do espectro médio com os picos detectados.
    Retorna o objeto Figure do Matplotlib e um dicionário com os picos detectados.
    """
    # Obter os IDs de grupos únicos
    unique_groups = unique(data.group_ids)

    # Inicializar a figura e o dicionário de picos
    fig, ax = plt.subplots(figsize=(10, 6))
    peaks = {}

    # Definir a largura para a transformação wavelet
    widths = arange(1, 10)

    # Loop sobre cada grupo
    for group_id in unique_groups:
        # Seleciona os espectros pertencentes a esse grupo
        sel = data.group_ids == group_id

        # Calcula o espectro médio para o grupo
        mean_spectrum = data.abss[sel, :].mean(axis=0)

        # Plota o espectro médio
        ax.plot(data.wn, mean_spectrum, label=f'Grupo {group_id}')

        # Detecta picos no espectro médio
        peak_indices = find_peaks_cwt(mean_spectrum, widths)
        group_name = str(data.args).split('::')[int(group_id - 1)]
        ax.text(0.05, 0.1 - unique_groups.tolist().index(group_id)*0.05,
                f'Grupo {group_name}: {len(peak_indices)} picos',
                transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')

        # Armazenar os picos detectados e plotar
        for j in peak_indices:
            peaks[(data.wn[j], mean_spectrum[j], int(group_id))] = mean_spectrum[j]
            ax.plot(data.wn[j], mean_spectrum[j], 'ro', markersize=8)

    # Configurar a legenda
    legenda = str(data.args).split('::')
    ax.legend(legenda)

    # Configurações dos eixos
    ax.set_xlabel('Número de onda (cm^{-1})')
    ax.set_ylabel('Absorbância')

    # Retorna a figura e o dicionário de picos
    return fig, peaks


def mplot_peaks(data: SpectroscopyData):
    """
    Função para plotar o espectro médio com os picos detectados.
    """
    _, _ = mplot_peaks_fig(data)
    plt.show()


def aplot(data: SpectroscopyData):
    """
    Função para plotar todos os espectros
    """
    unique_groups = unique(data.group_ids)
    for i in unique_groups:
        sel = data.group_ids == i
        d = data.abss[sel, :]
        color = data.colors[sel][0]
        for j in range(d.shape[0]):
            plt.plot(data.wn, d[j, :], color=color)
    legenda = str(data.args)
    legenda = legenda.split('::')
    plt.legend(legenda)
    plt.xlabel('numero de onda (cm^{-1})')
    plt.show()


__all__ = ["mplot", "aplot"]
