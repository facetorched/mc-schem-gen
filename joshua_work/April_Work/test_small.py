from mcschematic_plus import MCSchematicPlus
import numpy as np

# Test with a small subset of the data
cell_data = np.load("joshua_work/cell_input/martini2_cell.npy")
print(f"Data shape: {cell_data.shape}")

# Take a small 10x10x10 slice for testing
voxels = cell_data[0, :10, :10, :10, 0]  # Small slice, first channel
voxels = np.transpose(voxels, (2, 1, 0))  # Convert to (x, y, z)
print(f"Voxels shape: {voxels.shape}")
print(f"Number of True voxels: {np.sum(voxels)}")

# Create schematic and place volume
schem = MCSchematicPlus()
schem.placeVolume(voxels, "minecraft:stone", placePosition=(0, 0, 0))

# Save
schem.save("output/test_small.schem")
print("Test schematic saved successfully!")