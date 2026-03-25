import numpy as np
import os
from typing import BinaryIO, Sequence
from mcschematic import MCSchematic, MCStructure, Version
from nbtlib.tag import *
from nbtlib import File, parse_nbt

class VolumeStructure(MCSchematic):
    def __init__(self, schematicToLoadPath_or_mcStructure: str | MCStructure = None, version: 'Version' = None):
        if isinstance(schematicToLoadPath_or_mcStructure, str):
            schematicToLoadPath = schematicToLoadPath_or_mcStructure
            if not os.path.isfile(schematicToLoadPath):
                raise FileNotFoundError(f"Schematic file not found: {schematicToLoadPath}")
            if schematicToLoadPath.endswith(".schem"):
                self._initFromFile(schematicToLoadPath)
            else:
                try:
                    import amulet
                except ImportError:
                    raise ImportError("To load non .schem files, please install the 'amulet-core' package")
                level = amulet.load_level(os.path.abspath(schematicToLoadPath))
                dim = "main"
                sel = level.bounds(dim)
                self._defaultInit()
                version = ("java", version.value) if version is not None else ("java", self.getLatestVersion().value)
                for x in range(sel.min_x, sel.max_x):
                    for y in range(sel.min_y, sel.max_y):
                        for z in range(sel.min_z, sel.max_z):
                            block, blockEntity = level.get_version_block(x, y, z, dim, version=version)
                            if block.namespaced_name != "minecraft:air":
                                self.setBlock((x, y, z), block.full_blockstate)
                level.close()
        else:
            super().__init__(schematicToLoadPath_or_mcStructure)

    def placeVolume(self, volume: np.ndarray, blockData: str | None, colors: np.ndarray | None = None, placePosition: tuple[int, int, int] = (0, 0, 0)):
        """Add blocks for every True voxel in volume. The volume shape is Minecraft (x,y,z)."""
        volume = np.asarray(volume, dtype=bool)
        positions = np.argwhere(volume)  # shape (N, 3) with columns [x, y, z]
        if positions.size == 0:
            return
        positions += np.array(placePosition)  # offset coordinates
        _colors = colors[volume] if colors is not None else None
        self.setBlocks(positions, blockData)
    
    def setBlocks(self, positions: np.ndarray, blockData: str | Sequence[str] | None):
        """Add blocks for every (x,y,z) in points."""
        positions = np.asarray(positions)
        if positions.size == 0:
            return
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be an array of shape (N, 3)")
        if blockData is None:
            for x, y, z in positions:
                pos = (int(x), int(y), int(z))
                if pos in self._structure._blockStates:
                    del self._structure._blockStates[pos]
                if pos in self._structure._blockEntities:
                    del self._structure._blockEntities[pos]
            return
        if isinstance(blockData, str):
            for x, y, z in positions:
                self.setBlock((int(x), int(y), int(z)), blockData) # TODO slightly inefficient but simple
        else:
            for (x, y, z), bd in zip(positions, blockData):
                self.setBlock((int(x), int(y), int(z)), bd)

    def getBlocks(self):
        """Return a dictionary of positions to block names."""
        block_dict : dict[tuple[int, int, int], str] = {}
        for pos in self._structure.getBlockStates().keys():
            block_dict[pos] = self.getBlockDataAt(pos)
        return block_dict
    
    def getBlockName(self, blockData: str | None) -> str | None:
        """Get the block name from blockData string. If blockData is None, return None."""
        if blockData is None:
            return None
        return blockData.split("[")[0]  # Remove properties if present

    def replaceBlocks(self, oldBlock: str, newBlock: str | None):
        """Replace all instances of oldBlock with newBlock. If newBlock is None, remove the block."""
        block_states_copy = dict(self._structure._blockStates)  # avoid modifying dict during iteration
        if newBlock is None:
            for pos, blockPaletteId in block_states_copy.items():
                if self._structure._blockPalette[blockPaletteId] == oldBlock:
                    if pos in self._structure._blockEntities:
                        del self._structure._blockEntities[pos]
                    del self._structure._blockStates[pos]
            # remove from palette
            blockPaletteId = self._structure._blockPalette[oldBlock]
            del self._structure._blockPalette[oldBlock]
            del self._structure._blockPalette[blockPaletteId]  
        
        else:
            palette_copy = dict(self._structure._blockPalette)  # avoid modifying dict during iteration
            # change the palette entry
            blockPaletteId = self._structure._blockPalette[oldBlock]
            self._structure._blockPalette[blockPaletteId] = newBlock
            self._structure._blockPalette[newBlock] = self._structure._blockPalette.pop(oldBlock)

            for pos, blockPaletteId in block_states_copy.items():
                if palette_copy[blockPaletteId] == oldBlock:
                    if pos in self._structure._blockEntities:
                        del self._structure._blockEntities[pos]
                    self.setBlock(pos, newBlock)

    def splitByBlock(self):
        """Return a dictionary of block names and corresponding VolumeStructure objects."""
        block_dict : dict[str, VolumeStructure] = {}
        for pos, blockPaletteId in self._structure.getBlockStates().items():
            block_name = self._structure._blockPalette[blockPaletteId]
            if block_name not in block_dict:
                block_dict[block_name] = VolumeStructure()
            block_dict[block_name].setBlock(pos, block_name) # TODO slightly inefficient but simple
        return block_dict
    
    def getLatestVersion(self) -> 'Version':
        """Get the latest supported Minecraft version."""
        return max(Version, key=lambda v: v.value)

    def save(self, filepath: str | BinaryIO, version: 'Version' = None, fastSaving: bool = False):
        """Save the structure as a Sponge schematic file at `filepath`."""
        # if filepath is a string, ensure directory exists
        if isinstance(filepath, (str, os.PathLike)):
            filepath = str(filepath)
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            # add .schem extension if not present
            if not filepath.lower().endswith('.schem'):
                filepath += '.schem'
        if version is None:
            version = self.getLatestVersion()
        #################### Start copied from MCSchematic #####################
        ## Setup
        schemBounds = self._structure.getBounds()
        schemDims = self._structure.getStructureDimensions(schemBounds)
        # The vector amount by which minBounds is offsetted from 0 0 0
        schemOffset = schemBounds[0]

        ## BLOCK PALETTE
        ## We're doing the block palette early because it's gonna
        ## be useful in knowing which algorithm to use in when saving
        ## the blocks to the schematic
        # Get a cleaner version of the block palette, without the duplicates used
        # for back and forth O(1) access
        self._structure.getBlockPalette()
        cleanBlockPalette = self._structure.getBlockPalette()


        ## BLOCK DATA
        encodedBlockStates = self._getEncodedBlockStates(len(cleanBlockPalette),
                                                         schemDims,
                                                         schemOffset,
                                                         fastSaving)


        ## BLOCK ENTITIES
        blockEntitiesCompounds = \
            [
                self._blockEntityStringToSchemCompound(
                    (
                        position[0] - schemOffset[0],
                        position[1] - schemOffset[1],
                        position[2] - schemOffset[2]
                    ), blockEntityStr
                )
                for position, blockEntityStr in self._structure.getBlockEntities().items()
            ]


        ## Generate the schematic file from the byte array
        ## From Fearless's code which was taken from someone else lOl
        schematic = File({

            'Version': Int(2),
            'DataVersion': Int(version.value),
            'Metadata': Compound({
                'WEOffsetX': Int(schemOffset[0]),
                'WEOffsetY': Int(schemOffset[1]),
                'WEOffsetZ': Int(schemOffset[2]),
                'MCSchematicMetadata': Compound({
                    'Generated': String("Generated with love using Sloimay's MCSchematic Python Library, "
                                        "itself dependant on Valentin Berlier's nbtlib library.")
                })
            }),

            'Height': Short(schemDims[1]),
            'Length': Short(schemDims[2]),
            'Width': Short(schemDims[0]),

            'PaletteMax': Int(len(cleanBlockPalette)),
            'Palette': Compound(cleanBlockPalette),
            'BlockData': ByteArray(encodedBlockStates),

            'BlockEntities': List(blockEntitiesCompounds),

        }, gzipped=True, root_name='Schematic')
        #################### End copied from MCSchematic #####################

        schematic.save(filepath)
        

    def saveNBT(self, filepath: str, version : 'Version' = None, max_size: int | tuple[int, int, int] | None = None, filename_mode: str = "auto"):
        """
        Save the structure as one or more Minecraft schematic .nbt files in <directory>.
        If the structure exceeds max_size in any dimension, it will be split into multiple files.

        Parameters
        ----------
        filepath : str
            Directory to save the .nbt files. If multiple files are created, they will be named
            <filepath>_x_y_z.nbt where x,y,z are the tile indices.
        version : 'Version', optional
            The Version to include in the NBT file. If None, the latest version will be used.
        max_size : int | tuple | None, optional
            Maximum size in each dimension for a single .nbt file. If None, a single file will be created. Default is None.
        filename_mode : str, optional
            "auto" (default): use base_name.nbt if only one file is needed, otherwise use indexed names.
            "indexed": always use indexed names.
        """
        # ensure directory exists
        directory = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        if base_name.lower().endswith('.nbt'):
            base_name = base_name[:-4]
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if version is None:
            version = self.getLatestVersion()
        x_min, y_min, z_min = self._structure.getBounds()[0]
        x_max, y_max, z_max = self._structure.getBounds()[1]
        size_x, size_y, size_z = x_max - x_min, y_max - y_min, z_max - z_min
        if max_size is None:
            max_size = (size_x, size_y, size_z)
        elif isinstance(max_size, int):
            max_size = (max_size, max_size, max_size)
        nx = (size_x + max_size[0] - 1) // max_size[0]
        ny = (size_y + max_size[1] - 1) // max_size[1]
        nz = (size_z + max_size[2] - 1) // max_size[2]

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x0, y0, z0 = ix*max_size[0] + x_min, iy*max_size[1] + y_min, iz*max_size[2] + z_min
                    x1, y1, z1 = min(x0+max_size[0], x_max), min(y0+max_size[1], y_max), min(z0+max_size[2], z_max)
                    tile_size = (x1-x0, y1-y0, z1-z0)

                    root = Compound()
                    root["DataVersion"] = Int(version.value)
                    root["size"] = List[Int]([Int(tile_size[0]), Int(tile_size[1]), Int(tile_size[2])])

                    # Palette
                    palette = List[Compound]()
                    palette_index = {}
                    # Always include air
                    palette.append(Compound({"Name": String("minecraft:air")}))
                    palette_index["minecraft:air"] = 0
                    next_index = 1

                    # Blocks
                    blocks = List[Compound]()
                    
                    for (x,y,z), block in self.getBlocks().items():
                        if x0 <= x < x1 and y0 <= y < y1 and z0 <= z < z1:
                            rel = (x-x0, y-y0, z-z0)
                            
                            if block not in palette_index:
                                block_name = block.split("[")[0]
                                entry = Compound({"Name": String(block_name)})
                                if block.find("[") != -1:
                                    props_str = block[block.find("[")+1:block.find("]")]
                                    props = Compound()
                                    for prop in props_str.split(","):
                                        if "=" in prop: # Safety check for malformed strings
                                            key, value = prop.split("=")
                                            props[key] = String(value)
                                    entry["Properties"] = props
                                palette.append(entry)
                                palette_index[block] = next_index
                                next_index += 1
                            
                            state = palette_index[block]
                            btag = Compound()
                            btag["state"] = Int(state)
                            btag["pos"] = List[Int]([Int(rel[0]), Int(rel[1]), Int(rel[2])])
                            
                            if (x,y,z) in self._structure._blockEntities:
                                blockEntityString = self._structure._blockEntities[(x,y,z)]
                                if "{" in blockEntityString:
                                    nbtPortion = blockEntityString[blockEntityString.find("{"):] 
                                    btag["nbt"] = parse_nbt(nbtPortion)
                                    
                            blocks.append(btag)

                    root["palette"] = palette
                    root["blocks"] = blocks
                    root["entities"] = List[Compound]() # TODO: Add support for entities

                    # Save
                    if nx>1 or ny>1 or nz>1 or filename_mode == "indexed":
                        fname = f"{base_name}_{ix}_{iy}_{iz}.nbt"
                    elif filename_mode == "auto":
                        fname = f"{base_name}.nbt"
                    path = os.path.join(directory, fname)
                    
                    nbt_file = File(root, gzipped=True, root_name="Schematic")
                    nbt_file.save(path)
                    print(f"Saved {path} ({tile_size})")