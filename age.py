from dataclasses import dataclass
from glob import glob

import numpy as np


@dataclass
class SpectroscopyData:
    abss: np.ndarray
    wn: np.ndarray
    groups: np.ndarray
    args: np.ndarray


def read_file(file_path: str):
    file_contents  = np.loadtxt(file_path)
    # x
    wn = file_contents[:, 0]
    # y
    abss = file_contents[:, 1]
    data = np.column_stack((wn, abss))
    return np.flipud(data)

def read_dir(dir_path: str, group: str) -> SpectroscopyData:
    args = np.array(group)
    r = []
    file_paths = glob(f"{dir_path}/*.dpt")
    if not file_paths:
        raise FileNotFoundError(f"No files found in {dir_path}")
    file_contents = None
    for file_path in file_paths: 
        file_contents = read_file(file_path)
        r.append(file_contents[:, 1])

    if file_contents is None:
        raise ValueError("File contents is None")

    r = np.array(r)
    return SpectroscopyData(abss=r, wn=file_contents[:, 0], groups=args, args=args)

__all__ = ['read_dir', 'read_file', 'SpectroscopyData']
