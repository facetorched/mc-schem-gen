**mc-schem-gen** is a library for writing Minecraft schematics (.schem) and structures (.nbt) from ndarray or image data.

# Installing
Clone this repository into a local directory.
```sh
git clone https://github.com/facetorched/mc-schem-gen.git
```
Navagate into the repository and install the package locally. Optionally making it editable by including the `-e` flag.
```sh
pip install -e .
```

# Usage
The class `VolumeStructure` offers the main functionality of the package.

```python
from mc_schem_gen import VolumeStructure, read_tiff

vs = VolumeStructure()
# Load and add a layer from a TIFF image
vs.add_layer(read_tiff("tests/data/blobs.tiff"), "minecraft:blue_stained_glass")
# Load and add a layer from an existing schematic file
vs.add_schem("tests/data/min_cell.schematic")
# Save both a schematic and structure files
vs.save_schem("output/example.schem")
vs.save_nbt("output/example", "structure")

```
