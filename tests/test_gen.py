import numpy as np
from mc_schem_gen import VolumeStructure, read_tiff
from amulet_nbt import load

OUTPUT_DIR = "tests/output/"

def test_small_volume():
    vol = np.zeros((10, 10, 10), dtype=bool)
    vol[1:5, 1:5, 1:5] = True
    vs = VolumeStructure()
    vs.add_layer(vol, "minecraft:stone")
    vs.save_nbt(f"{OUTPUT_DIR}/small_struct", "structure")
    vs.save_schem(f"{OUTPUT_DIR}/small_struct.schem")
    print("NBT file(s) saved to ./output")
    # try to load the nbt
    load(f"{OUTPUT_DIR}/small_struct/structure_0_0_0.nbt")
    print("Loaded NBT")

def test_large_volume():
    vol = np.zeros((100, 100, 100), dtype=bool)
    vol[10:90, 10:90, 10:90] = True
    vs = VolumeStructure()
    vs.add_layer(vol, "minecraft:dirt")
    vs.save_nbt(f"{OUTPUT_DIR}/large_struct", "structure")
    vs.save_schem(f"{OUTPUT_DIR}/large_schem.schem")
    print("Large NBT file(s) saved to ./output")
    # try to load one of the nbt files
    load(f"{OUTPUT_DIR}/large_struct/structure_0_0_0.nbt")
    print("Loaded large NBT")

def test_tiff():
    vol = read_tiff("tests/data/blobs.tiff")
    vs = VolumeStructure()
    vs.add_layer(vol, "minecraft:blue_stained_glass")
    vs.save_nbt(f"{OUTPUT_DIR}/tiff_struct", "structure")
    vs.save_schem(f"{OUTPUT_DIR}/tiff_schem.schem")
    print("TIFF NBT file(s) saved to ./output")
    # try to load one of the nbt files
    load(f"{OUTPUT_DIR}/tiff_struct/structure_0_0_0.nbt")
    print("Loaded TIFF NBT")

def test_schem():
    vs = VolumeStructure()
    vs.add_schem("tests/data/min_cell.schematic")
    vs.save_nbt(f"{OUTPUT_DIR}/schem_struct", "structure")
    vs.save_schem(f"{OUTPUT_DIR}/schem_struct.schem")
    print("Schem NBT file(s) saved to ./output")
    # try to load one of the nbt files
    load(f"{OUTPUT_DIR}/schem_struct/structure_0_0_0.nbt")
    print("Loaded Schem NBT")

if __name__ == "__main__":
    test_small_volume()
    test_large_volume()
    test_tiff()
    test_schem()
