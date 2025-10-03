import numpy as np
import tifffile
import os

from amulet.api.block import Block
import amulet
from amulet_nbt import NamedTag, CompoundTag, ListTag, IntTag, StringTag


# --------------------------
# Volume loading utilities
# --------------------------

def _to_bool_volume(arr: np.ndarray) -> np.ndarray:
    """Convert raw array in ImageJ format z, y, x to boolean volume (x, z, y) = Minecraft (x, y, z) orientation."""
    if arr.ndim == 4:  # drop color channels
        arr = arr.sum(axis=-1)
    # rearrange axes to (x, z, y) to be compatible with Minecraft.
    # The first axis is West->East, the second is Bottom->Top, and the last is North->South
    arr = np.moveaxis(arr, 2, 0)
    return arr.astype(bool)

def read_tiff(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    return _to_bool_volume(img)

def read_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    return _to_bool_volume(arr)


# --------------------------
# VolumeStructure class
# --------------------------

class VolumeStructure:
    def __init__(self, platform: str = "java", version: tuple = (1, 21, 8)):
        self.platform = platform
        self.version = version
        self._blocks = {}   # (x,y,z) -> Block
        self._max_x = self._max_y = self._max_z = -1
    
    def _update_size(self, x: int, y: int, z: int):
        self._max_x = max(self._max_x, x)
        self._max_y = max(self._max_y, y)
        self._max_z = max(self._max_z, z)

    def set_block(self, x: int, y: int, z: int, block: Block):
        self._blocks[(x, y, z)] = block
        self._update_size(x, y, z)
    
    def get_block(self, x: int, y: int, z: int) -> Block:
        return self._blocks.get((x, y, z), Block("minecraft", "air"))
    
    def get_blocks(self):
        return self._blocks.items()
    
    def get_volume_size(self):
        return (self._max_x + 1, self._max_y + 1, self._max_z + 1)

    def add_layer(self, volume: np.ndarray, block_spec: str):
        """Add blocks for every True voxel in volume."""
        ns, base = block_spec.split(":", 1)
        block_obj = Block(ns, base)
        it = np.ndindex(volume.shape)
        for x, y, z in it:
            if volume[x, y, z]:
                self.set_block(x, y, z, block_obj)

    def add_schem(self, path: str):
        """
        Load an existing Sponge schematic (.schem) file and merge its blocks
        into this VolumeStructure. Useful for converting schem -> nbt.
        """
        level = amulet.load_level(path)
        try:
            # Sponge schematics always use a "main" dimension
            dim = "main"
            # iterate all blocks in the level’s selection box
            sel = level.bounds(dim)
            for x in range(sel.min_x, sel.max_x):
                for y in range(sel.min_y, sel.max_y):
                    for z in range(sel.min_z, sel.max_z):
                        block, _ = level.get_version_block(x, y, z, dim, (self.platform, self.version))
                        if block.namespaced_name != "minecraft:air":
                            self.set_block(x, y, z, block)
        finally:
            level.close()

    def split_by_block(self):
        """Return a dictionary of block names and corresponding VolumeStructure objects."""
        block_dict = {}
        for (x, y, z), block in self._blocks.items():
            block_name = block.namespaced_name
            if block_name not in block_dict:
                block_dict[block_name] = VolumeStructure(self.platform, self.version)
            block_dict[block_name].set_block(x, y, z, block)
        return block_dict

    # --------------------------
    # Save as Sponge schematic
    # --------------------------
    def save_schem(self, filepath: str):
        size_x, size_y, size_z = self.get_volume_size()
        wrapper = amulet.level.formats.sponge_schem.SpongeSchemFormatWrapper(filepath)
        bounds = amulet.api.selection.SelectionGroup(
            amulet.api.selection.SelectionBox((0,0,0), (size_x, size_y, size_z))
        )
        wrapper.create_and_open(self.platform, self.version, bounds, overwrite=True)
        wrapper.save()
        wrapper.close()
        level = amulet.load_level(filepath)
        try:
            dim = "main"
            game_ver = (self.platform, self.version)
            for (x, y, z), block in self._blocks.items():
                level.set_version_block(x, y, z, dim, game_ver, block)
            level.save()
        finally:
            level.close()

    # --------------------------
    # Save as structure NBT(s)
    # --------------------------
    def save_nbt(self, directory: str, base_name: str, dataversion: int = None, max_size: int = 48):
        os.makedirs(directory, exist_ok=True)
        size_x, size_y, size_z = self.get_volume_size()
        nx = (size_x + max_size - 1) // max_size
        ny = (size_y + max_size - 1) // max_size
        nz = (size_z + max_size - 1) // max_size

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x0, y0, z0 = ix*max_size, iy*max_size, iz*max_size
                    x1, y1, z1 = min(x0+max_size, size_x), min(y0+max_size, size_y), min(z0+max_size, size_z)
                    tile_size = (x1-x0, y1-y0, z1-z0)

                    root = CompoundTag()
                    if dataversion:
                        root["DataVersion"] = IntTag(dataversion)
                    root["size"] = ListTag([IntTag(tile_size[0]), IntTag(tile_size[1]), IntTag(tile_size[2])])

                    # Palette
                    palette = ListTag()
                    palette_index = {}
                    # Always include air
                    palette.append(CompoundTag({"Name": StringTag("minecraft:air")}))
                    palette_index["minecraft:air"] = 0
                    next_index = 1

                    # Blocks
                    blocks = ListTag()
                    for (x,y,z), block in self._blocks.items():
                        if x0 <= x < x1 and y0 <= y < y1 and z0 <= z < z1:
                            rel = (x-x0, y-y0, z-z0)
                            name = block.namespaced_name
                            if name not in palette_index:
                                entry = CompoundTag({"Name": StringTag(name)})
                                if block.properties:
                                    props = CompoundTag()
                                    for k,v in block.properties.items():
                                        props[k] = StringTag(str(v))
                                    entry["Properties"] = props
                                palette.append(entry)
                                palette_index[name] = next_index
                                next_index += 1
                            state = palette_index[name]
                            btag = CompoundTag()
                            btag["state"] = IntTag(state)
                            btag["pos"] = ListTag([IntTag(rel[0]), IntTag(rel[1]), IntTag(rel[2])])
                            blocks.append(btag)

                    root["palette"] = palette
                    root["blocks"] = blocks
                    root["entities"] = ListTag()

                    # Save
                    if nx>1 or ny>1 or nz>1 or True:
                        fname = f"{base_name}_{ix}_{iy}_{iz}.nbt"
                    else:
                        fname = f"{base_name}.nbt"
                    path = os.path.join(directory, fname)
                    NamedTag(root, name="").save_to(path, compressed=True, little_endian=False)
                    print(f"Saved {path} ({tile_size})")
