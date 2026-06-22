from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh, read_npy
from mcschematic import Version

schem = MCSchematicPlus()

INPUT_FILE = "joshua_work/cell_input/martini3.schem"
CELL_NAME = "martini3"
OUTPUT_PATH = f"joshua_work/output/{CELL_NAME}"

# Load blocks from an existing schematic file
schem.placeSchematic(MCSchematicPlus(INPUT_FILE), placePosition=(0,0,0))

# Save nbt file
schem.saveNBT(f"{OUTPUT_PATH}/{CELL_NAME}.nbt", Version.JE_1_20_1, maxSize=(48, 48, 48), shifted=False)

# Save individual schematics for each block type
# split = schem.splitByBlock()
# for block_namespaced_name, block_vs in split.items():

#     block_name = block_namespaced_name.replace("minecraft:", "")

#     block_vs.saveNBT(f"{OUTPUT_PATH}/{block_name}/v.nbt", Version.JE_1_20_1, maxSize=(48, 48, 48), removeAir=False, shifted=False, rewriteExisting=True)


# Run: .venv/Scripts/python.exe joshua_work/gen_schem_splitter.py