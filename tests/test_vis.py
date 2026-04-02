from mcschematic_plus import MCSchematicPlus

def test_show():
    import pyvista as pv
    
    p = pv.Plotter()
    p.add_mesh(pv.Sphere(radius=0.5), color="red", name="origin_marker")
    schem = MCSchematicPlus()
    schem.placeSchematic(MCSchematicPlus("tests/data/mesh_f.schem"), placePosition=(0, 0, -41))
    schem.show(plotter=p)

if __name__ == "__main__":
    test_show()
    print("All tests passed.")