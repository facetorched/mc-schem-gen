import csv
import numpy as np
from scipy.spatial import KDTree
from numpy.typing import ArrayLike
from importlib.resources import files

_INTERNAL_COLORMAPS = {}

def get_internal_colormap(name: str) -> 'BlockColormap':
    """
    Returns a BlockColormap object for one of the provided colormaps.
    Loads the colormap from the package data on first request and caches it for future use.

    Parameters
    ----------
    name : str 
        Name of the colormap. Options are 'standard', 'all', 'smooth'.

    Returns
    -------
    colormap : BlockColormap
        The requested BlockColormap object.
    """
    if name not in _INTERNAL_COLORMAPS:
        _INTERNAL_COLORMAPS[name] = BlockColormap(files("mc_schem_gen").joinpath(f"data/block_colormaps/{name}.csv"))
    return _INTERNAL_COLORMAPS[name]


class BlockColormap:
    def __init__(self, csv_path):
        bs : list[str] = []
        cs : list[tuple[int, int, int]] = []
        alphas : list[int] = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            block_state_idx = header.index("block_state")
            r_idx = header.index("r")
            g_idx = header.index("g")
            b_idx = header.index("b")
            a_idx = header.index("a")
            for row in reader:
                block_state = row[block_state_idx]
                r = int(float(row[r_idx]))
                g = int(float(row[g_idx]))
                b = int(float(row[b_idx]))
                a = int(float(row[a_idx]))
                bs.append(block_state)
                cs.append((r, g, b))
                alphas.append(a)
        self._block_states = np.array(bs)
        self._colors = np.array(cs)
        self._tree = KDTree(self._colors)
        self._alphas = np.array(alphas) # for now we don't query based on alpha
        self._map_cache = {}

    def get_block(self, rgb : ArrayLike) -> str:
        """
        Returns the closest matching block_state for the given (R, G, B) color or array of colors.
        """
        if rgb in self._map_cache:
            return self._map_cache[rgb]
        _dist, index = self._tree.query(rgb, k=1)
        self._map_cache[rgb] = self._block_states[index]
        return self._block_states[index]
    
    def get_color(self, block_state: str) -> np.ndarray:
        """
        Returns the (R, G, B) color for the given block_state.
        """
        index = np.where(self._block_states == block_state)[0]
        if len(index) == 0:
            raise ValueError(f"Block state {block_state} not found in colormap.")
        return self._colors[index[0]]
    
    def get_alpha(self, block_state: str) -> int:
        """
        Returns the alpha value for the given block_state.
        """
        index = np.where(self._block_states == block_state)[0]
        if len(index) == 0:
            raise ValueError(f"Block state {block_state} not found in colormap.")
        return self._alphas[index[0]]

    @property
    def block_states(self) -> np.ndarray:
        return self._block_states
    
    @property
    def colors(self) -> np.ndarray:
        return self._colors
    
    @property
    def alphas(self) -> np.ndarray:
        return self._alphas