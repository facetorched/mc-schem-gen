"""
mc-schem-gen: Generate Minecraft .schem and .nbt structures from 3D images.
"""

from .volume_structure import VolumeStructure
from .data_loaders import (
    to_mc_bool_volume,
    read_tiff,
    read_npy,
    read_mesh,
    voxelize_mesh,
)
