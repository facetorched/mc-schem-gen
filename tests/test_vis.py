from mcschematic_plus import MCSchematicPlus, read_mesh
import pyvista as pv

def test_show():
    p = pv.Plotter()
    p.add_mesh(pv.Sphere(radius=0.5), color="red", name="origin_marker")
    schem = MCSchematicPlus()
    schem.placeSchematic(MCSchematicPlus("tests/data/mesh_f.schem"), placePosition=(0, 0, -41))
    schem.show(plotter=p)

def test_show_color():
    p = pv.Plotter()
    p.add_mesh(pv.Sphere(radius=0.5), color="red", name="origin_marker")
    schem = MCSchematicPlus()
    schem.placeSchematic(MCSchematicPlus("tests/data/mesh_glycine.schem"), placePosition=(0, 0, 0))
    schem.show(plotter=p)

if __name__ == "__main__":
    test_show()
    test_show_color()
    print("All tests passed.")