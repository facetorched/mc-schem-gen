from turtle import position

from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh, read_npy
from mcschematic import Version

schem = MCSchematicPlus()

# # Load 3D data from a multipage tiff
# schem.placeVolume(read_tiff("../tests/data/blobs.tiff"), "minecraft:blue_stained_glass")

# Load blocks from an existing schematic file
schem.placeSchematic(MCSchematicPlus("joshua_work/cell_input/martini2_cell.schem"), placePosition=(0,0,0))

# # Load a 3D model

# Visualize the schematic (commented out as it fails on empty meshes)
# schem.show()

# Save both a schematic and nbt file
# schem.save("joshua_work/output/martini2.schem")
# schem.saveNBT("joshua_work/output/martini2.nbt", Version.JE_1_20_1, maxSize=(48, 48, 48))

# Save individual schematics for each block type
split = schem.splitByBlock()
for block_namespaced_name, block_vs in split.items():
    block_name = block_namespaced_name.replace(":", "_")
    block_vs.saveNBT(f"joshua_work/output/martini3/{block_name}/v.nbt", Version.JE_1_20_1, maxSize=(48, 48, 48))
    # block_vs.save_schem(f"joshua_work/output/martini2/split_{block_name}.schem")