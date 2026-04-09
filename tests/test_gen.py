import numpy as np
from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh
from nbtlib import load

OUTPUT_DIR = "tests/output"

def test_small_volume():
    vol = np.zeros((10, 10, 10), dtype=bool)
    vol[1:5, 1:5, 1:5] = True
    schem = MCSchematicPlus()
    schem.placeVolume(vol, "minecraft:stone")
    schem.saveNBT(f"{OUTPUT_DIR}/small.nbt")
    schem.save(f"{OUTPUT_DIR}/small.schem")
    # try to load the nbt
    load(f"{OUTPUT_DIR}/small.nbt")

def test_large_volume():
    vol = np.zeros((100, 100, 100), dtype=bool)
    vol[10:90, 10:90, 10:90] = True
    schem = MCSchematicPlus()
    schem.placeVolume(vol, "minecraft:dirt")
    schem.saveNBT(f"{OUTPUT_DIR}/large.nbt", maxSize=48)
    schem.save(f"{OUTPUT_DIR}/large.schem")
    # try to load the nbt
    load(f"{OUTPUT_DIR}/large_0_0_0.nbt")

def test_tiff():
    vol = read_tiff("tests/data/blobs.tiff")
    schem = MCSchematicPlus()
    schem.placeVolume(vol, "minecraft:blue_stained_glass")
    schem.saveNBT(f"{OUTPUT_DIR}/tiff_blobs.nbt")
    schem.save(f"{OUTPUT_DIR}/tiff_blobs.schem")

def test_split():
    dirt_vol = np.zeros((10, 10, 10), dtype=bool)
    dirt_vol[1:5, 1:5, 1:5] = True
    stone_vol = np.zeros((10, 10, 10), dtype=bool)
    stone_vol[5:9, 5:9, 5:9] = True
    schem = MCSchematicPlus()
    schem.placeVolume(dirt_vol, "minecraft:dirt")
    schem.placeVolume(stone_vol, "minecraft:stone")
    split = schem.splitByBlock()
    assert "minecraft:dirt" in split
    assert "minecraft:stone" in split
    assert len(split) == 2
    # save the split structures
    for block_namespaced_name, block_vs in split.items():
        block_name = block_namespaced_name.replace(":", "_")
        block_vs.saveNBT(f"{OUTPUT_DIR}/split_{block_name}.nbt")
        block_vs.save(f"{OUTPUT_DIR}/split_{block_name}.schem")

def test_mesh():
    vol, pos, _ = read_mesh("tests/data/letter_F.vtk", spacing=0.1, minimum=(-2.0, -2.0, 0), origin=None)
    assert np.array_equal(pos, np.array([0, 0, -40]))
    schem = MCSchematicPlus()
    schem.placeVolume(vol, "minecraft:blue_stained_glass")
    schem.saveNBT(f"{OUTPUT_DIR}/mesh_f.nbt")
    schem.save(f"{OUTPUT_DIR}/mesh_f.schem")
    vol2, pos2, _ = read_mesh("tests/data/letter_F.vtk", spacing=0.1, minimum=(-2.0, -2.0, 0), origin=(0, 0, 0))
    assert np.array_equal(pos2, np.array([-20, 0, -20]))
    schem2 = MCSchematicPlus()
    schem2.placeVolume(vol2, "minecraft:blue_stained_glass", placePosition=pos2)
    schem2.saveNBT(f"{OUTPUT_DIR}/mesh_f_origin.nbt")
    schem2.save(f"{OUTPUT_DIR}/mesh_f_origin.schem")

def test_mesh_color():
    voxels, position, scalars = read_mesh("tests/data/glycine.glb", spacing=0.1, edge_mode="inner", compute_scalars=True)
    colors = (scalars * 255).astype(int)
    schem = MCSchematicPlus()
    schem.placeVolume(voxels, colors, blockColormap="standard", placePosition=position)
    schem.saveNBT(f"{OUTPUT_DIR}/mesh_glycine.nbt")
    schem.save(f"{OUTPUT_DIR}/mesh_glycine.schem")

def replace_test():
    vol = np.zeros((10, 10, 10), dtype=bool)
    vol[1:5, 1:5, 1:5] = True
    schem = MCSchematicPlus()
    schem.placeVolume(vol, "minecraft:stone")
    schem.replaceBlocks("minecraft:stone", "minecraft:dirt")
    blocks = schem.getBlocks()
    assert all(block == "minecraft:dirt" for _, block in blocks.items())
    schem.replaceBlocks("minecraft:dirt", None)
    blocks = schem.getBlocks()
    assert len(blocks) == 0

def test_schem():
    schem = MCSchematicPlus("tests/data/min_cell.schem")
    schem.saveNBT(f"{OUTPUT_DIR}/schem_cell.nbt")
    schem.save(f"{OUTPUT_DIR}/schem_cell.schem")

def test_mcedit_schem():
    try:
        import amulet
    except ImportError:
        print("Amulet not installed, skipping test_mcedit_schem")
        return
    schem = MCSchematicPlus("tests/data/min_cell.schematic")
    schem.saveNBT(f"{OUTPUT_DIR}/mceditschem_cell.nbt")
    schem.save(f"{OUTPUT_DIR}/mceditschem_cell.schem")

if __name__ == "__main__":
    test_small_volume()
    test_large_volume()
    test_tiff()
    test_split()
    test_mesh()
    test_mesh_color()
    replace_test()
    test_schem()
    test_mcedit_schem()
    print("All tests passed.")
