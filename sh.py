import matplotlib.pyplot as plt
from numpy import unique

from read import SpectroscopyData


def mplot(data: SpectroscopyData):
    """
    Função para plotar o espectro medio
    """
    unique_groups = unique(data.group_ids)
    for i in unique_groups:
        sel = data.group_ids == i
        plt.plot(data.wn, data.abss[sel,:].mean(0))
    legenda = str(data.args)
    legenda = legenda.split('::')
    plt.legend(legenda)
    plt.xlabel('numero de onda (cm^{-1})')
    plt.show()

def aplot(data: SpectroscopyData):
    """
    Função para plotar todos os espectros
    """
    unique_groups = unique(data.group_ids)
    for i in unique_groups:
        sel = data.group_ids == i
        d = data.abss[sel,:]
        color = data.colors[sel][0]
        for j in range(d.shape[0]):
            plt.plot(data.wn, d[j,:], color=color)
    legenda = str(data.args)
    legenda = legenda.split('::')
    plt.legend(legenda)
    plt.xlabel('numero de onda (cm^{-1})')
    plt.show()

__all__ = ["mplot", "aplot"]
