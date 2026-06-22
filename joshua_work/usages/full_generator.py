from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh, read_npy
from mcschematic import Version

schem = MCSchematicPlus()

INPUT_FILE = "joshua_work/cell_input/martini3.schem"
CELL_NAME = "martini3"
OUTPUT_PATH = f"joshua_work/output/{CELL_NAME}"

# Load blocks from an existing schematic file
schem.placeSchematic(MCSchematicPlus(INPUT_FILE), placePosition=(0,0,0))

# Save nbt file
schem.saveNBT(f"{OUTPUT_PATH}/{CELL_NAME}.nbt", Version.JE_1_21_5, shifted=False)


# Run: .venv/Scripts/python.exe joshua_work/usages/full_generator.py