from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh, read_npy
from mcschematic import Version

schem = MCSchematicPlus()

ER_INPUT = "joshua_work/cell_input/yeast/yeast_er.schem"
CELL_NAME = "yeast2"
OUTPUT_PATH = f"joshua_work/output/{CELL_NAME}"



for i in range(30, 50):
    schem.placeSchematic(MCSchematicPlus(ER_INPUT), placePosition=(-11,-30,-17))

    input_path = f"joshua_work/cell_input/yeast/yeast_g2_diffusion/yeast_g2_{i:04d}.schem"

    # Load blocks from an existing schematic file
    schem.placeSchematic(MCSchematicPlus(input_path), placePosition=(-11,-30,-17))

    # Save nbt file
    schem.saveNBT(f"{OUTPUT_PATH}/{CELL_NAME}/er_{i}.nbt", Version.JE_1_20_1, shifted=False)

# This was back when this was in the general folder
# .venv/Scripts/python.exe joshua_work/using_schem.py