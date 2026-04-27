from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh, read_npy
from mcschematic import Version

schem = MCSchematicPlus()

# # Load 3D data from a multipage tiff
# schem.placeVolume(read_tiff("../tests/data/blobs.tiff"), "minecraft:blue_stained_glass")

# Load blocks from an existing schematic file
# schem.placeSchematic(MCSchematicPlus("/cell_input/min_cell.schematic"), placePosition=(0,0,0))

# # Load a 3D model
import numpy as np
cell_data = np.load("joshua_work/cell_input/martini2_cell.npy")
# Process a smaller 50x50x50 region for testing - using channel 2 which has data
voxels = cell_data[0, 100:150, 100:150, 100:150, 2]  # Channel 2 has the most voxels. I'm assuming this goes: TIME, Z, Y, X, TYPE. 2 = cytosolic proteins
voxels = np.transpose(voxels, (2, 1, 0))  # Convert from (z, y, x) to (x, y, z) for Minecraft
position = (0, 0, 0)
schem.placeVolume(voxels, "minecraft:stone", placePosition=position)

voxels, position, scalars = read_mesh("tests/data/glycine.glb", spacing=0.1, edge_mode="inner", compute_scalars=True)
colors = (scalars * 255).astype(int)
voxels = cell_data[0, 100:150, 100:150, 100:150, 2]  # Channel 2 has the most voxels
# channel 0: 325,496
# channel 1: 197,050
# channel 2: 779,584
# channel 3: 89,881
# channel 4: 209,295
# channel 5: 191,865
voxels = np.transpose(voxels, (2, 1, 0))  # Convert from (z, y, x) to (x, y, z) for Minecraft
position = (0, 0, 0)
schem.placeVolume(voxels, colors, blockColormap="standard", placePosition=position)

# Trying to make this myself:
voxels = cell_data[0]

schem.placeVolume(voxels, colors, blockColormap="standard", placePosition=position)


# Visualize the schematic (commented out as it fails on empty meshes)
# schem.show()

# Save both a schematic and nbt file
schem.save("joshua_work/output/martini2.schem")
schem.saveNBT("joshua_work/output/martini2.nbt", Version.JE_1_20_1)

# Save individual schematics for each block type
# split = schem.split_by_block()
# for block_namespaced_name, block_vs in split.items():
#     block_name = block_namespaced_name.replace(":", "_")
#     block_vs.save_nbt(f"output/martini2_{block_name}", "structure")
#     # block_vs.save_schem(f"output/martini2/split_{block_name}.schem")