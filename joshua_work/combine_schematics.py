from mcschematic_plus import MCSchematicPlus
from mcschematic import Version
import os

# List of schematic files to combine
schematic_files = [
    "joshua_work/cell_input/martini2_cytosolic_proteins.schem",
    "joshua_work/cell_input/martini2_dna.schem",
    "joshua_work/cell_input/martini2_membrane.schem",
    "joshua_work/cell_input/martini2_membrane_proteins.schem",
    "joshua_work/cell_input/martini2_metabolites.schem",
    "joshua_work/cell_input/martini2_rna.schem"
]

# Create a new schematic to hold the combined result
combined_schem = MCSchematicPlus()

# Load and combine each schematic
for i, schem_file in enumerate(schematic_files):
    if os.path.exists(schem_file):
        print(f"Loading {schem_file}...")
        # Load the schematic
        loaded_schem = MCSchematicPlus(schem_file)
        # Place it into the combined schematic at the origin (they will overlap)
        combined_schem.placeSchematic(loaded_schem, placePosition=(0, 0, 0))
        print(f"Added {schem_file} to combined schematic")
    else:
        print(f"Warning: {schem_file} not found")

# Save the combined schematic
output_schem = "joshua_work/output/combined_cell.schem"
output_nbt = "joshua_work/output/combined_cell.nbt"

combined_schem.save(output_schem)
combined_schem.saveNBT(output_nbt, Version.JE_1_20_1)

print(f"Combined schematic saved to {output_schem}")
print(f"Combined NBT saved to {output_nbt}")
print("Combination complete!")