import numpy as np
from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh, read_npy
from mcschematic import Version

schem = MCSchematicPlus()
cell_data = np.load(r"C:\Users\public.QCB-MC-VR\Documents\GitHub\Kevin File converter\mcschematic-plus\joshua_work\cell_input\martini3.npy")

# Q What does cell_data look like?
# A: It looks like: [size 0 (timestep?)][x][y][z][molecule] = is "molecule" at (x,y,z)?

colors = {
    0 : "minecraft:lime_stained_glass",
    1 : "minecraft:blue_concrete",
    2 : "minecraft:red_concrete",
    3 : "minecraft:yellow_concrete",
    4 : "minecraft:cyan_concrete",
    5 : "minecraft:purple_concrete"
}

# create one single schematic containing all components
schem = MCSchematicPlus()
for i, component in enumerate(colors.keys()):
    vol = cell_data[..., i] # already bool
    # mask any previously placed components to avoid overlap
    for j in range(i):
        vol = vol & ~cell_data[..., j]
    schem.placeVolume(vol, colors[component])

schem.save("schematics/martini3.schem")

# Save both a schematic and nbt file
schem.save("joshua_work/output/martini2_new.schem")