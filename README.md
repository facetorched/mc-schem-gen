**mcschematic-plus** is a library that extends the functionality of [mcschematic](https://github.com/Sloimayyy/mcschematic) for creating Minecraft schematics (.schem) and structures (.nbt) from ndarray, image or 3D model data.

# Installing
Clone this repository into a local directory.
```sh
git clone https://github.com/facetorched/mcschematic-plus.git
```
Navagate into the repository and install the package locally. Optionally making it editable by including the `-e` flag.
```sh
pip install -e .
```

# Usage
The class `MCSchematicPlus` offers the main functionality of the package and is a drop-in replacement for `MCSchematic`.

```python
from mcschematic_plus import MCSchematicPlus, read_tiff, read_mesh

schem = MCSchematicPlus()

# Load 3D data from a multipage tiff
schem.placeVolume(read_tiff("tests/data/blobs.tiff"), "minecraft:blue_stained_glass")

# Load 2D RGB image
image = read_image("tests/data/qcb.png")
mask = image.sum(axis=-1).astype(bool)
schem.placeVolume(mask, image, blockColormap="standard")

# Load blocks from an existing schematic file
schem.placeSchematic(MCSchematicPlus("tests/data/min_cell.schematic"))

# Load a 3D model
voxels, position, scalars = read_mesh("tests/data/glycine.glb", spacing=0.1, edge_mode="inner", compute_scalars=True)
colors = (scalars * 255).astype(int)
schem.placeVolume(voxels, colors, blockColormap="standard", placePosition=position)

# Visualize the schematic
schem.show()

# Save both a schematic and nbt file
vs.save_schem("output/example.schem")
vs.save_nbt("output/example", "structure")

# Save individual schematics for each block type
split = vs.split_by_block()
for block_namespaced_name, block_vs in split.items():
    block_name = block_namespaced_name.replace(":", "_")
    block_vs.save_nbt(f"output/example_{block_name}", "structure")
    block_vs.save_schem(f"output/example/split_{block_name}.schem")
```
