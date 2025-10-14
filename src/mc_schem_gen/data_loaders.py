import numpy as np
import pyvista as pv
import tifffile
import trimesh
import mesh_to_sdf

def to_mc_bool_volume(arr: np.ndarray, true_value=None) -> np.ndarray:
    """
    Convert raw array in ImageJ format z, y, x to boolean volume (x, z, y) = Minecraft (x, y, z) orientation.
    
    Parameters
    ----------
    arr : ndarray
        Input array in ImageJ format (z, y, x) or (z, y, x, c). Color channels are summed if present.
    true_value : optional
        The value to be considered as True in the boolean volume. If None, any non-zero value is True.

    Returns
    -------
    result : ndarray
        Boolean volume in Minecraft orientation (x, z, y).
    """
    if arr.ndim == 4:  # drop color channels
        arr = arr.sum(axis=-1)
    # rearrange axes to (x, z, y) to be compatible with Minecraft.
    # The first axis is West->East, the second is Bottom->Top, and the last is North->South
    arr = np.moveaxis(arr, 2, 0)
    if true_value is not None:
        return arr == true_value
    return arr.astype(bool)

def read_tiff(path: str, true_value=None) -> np.ndarray:
    """
    Read a multi-page TIFF file and return a boolean volume.

    Parameters
    ----------
    path : str
        Path to the TIFF file.
    true_value : optional
        The value to be considered as True in the boolean volume. If None, any non-zero value is True.
    
    Returns
    -------
    result : ndarray
        3D binary mask translated to Minecraft coordinates where True indicates presence of structure.
    """
    return to_mc_bool_volume(tifffile.imread(path), true_value=true_value)

def read_npy(path: str, true_value=None) -> np.ndarray:
    """
    Read a NumPy .npy file and return a boolean volume.

    Parameters
    ----------
    path : str
        Path to the .npy file.
    true_value : optional
        The value to be considered as True in the boolean volume. If None, any non-zero value is True.

    Returns
    -------
    result : ndarray
        3D binary mask translated to Minecraft coordinates where True indicates presence of structure.
    """
    return to_mc_bool_volume(np.load(path), true_value=true_value)

def read_mesh(filename: str,
              spacing: float = 1.0,
              origin: tuple = None,
              size: tuple = None,
              ignore_clip=False,
              fill: bool = True,
              edge_mode: str = "center",
              method: str = "mesh_to_sdf"
              ) -> np.ndarray:
    """
    Load a mesh from any VTK-compatible file format and convert to a boolean voxel volume.

    Parameters
    ----------
    filename : str
        Path to the mesh file.
    spacing : float or (3,) tuple, optional
        Absolute voxel spacing in mesh coordinates.
    origin : (3,) tuple, optional
        Force a specific origin for lattice alignment. If None, uses mesh minimum bounds.
    size : (3,) tuple, optional
        Size of the voxel grid in mesh coordinates. If None, uses mesh bounds.
    ignore_clip : bool, optional
        If True, does not check if the mesh fits completely within the voxel grid. Default is False.
    fill : bool, optional
        If True, fill interior voxels. If False, only surface voxels. Default is True.
    edge_mode : {'center', 'inner', 'outer'}, optional
        'center' selects voxels whose centers are within the mesh surface,
        'inner' selects voxels whose centers are inside or touching the surface from inside,
        'outer' selects voxels whose centers are outside or touching the surface from outside.
    method : {'mesh_to_sdf', 'implicit_distance'}, optional
        Method to compute signed distance field. 'mesh_to_sdf' is more robust for complex meshes. 'implicit_distance' uses PyVista's built-in method.
    
    Returns
    -------
    result : ndarray
        3D binary mask translated to Minecraft coordinates where True indicates presence of structure.
    """
    return to_mc_bool_volume(
        voxelize_mesh(
            pv.read(filename),
            spacing=spacing,
            origin=origin,
            size=size,
            ignore_clip=ignore_clip,
            fill=fill,
            edge_mode=edge_mode,
            method=method
        ),
    )

def voxelize_mesh(mesh: pv.DataSet,
                   spacing: float = 1.0,
                   origin: tuple = None,
                   size: tuple = None,
                   ignore_clip=False,
                   fill: bool = True,
                   edge_mode: str = "center",
                   method: str = "mesh_to_sdf"
                   ) -> np.ndarray:
    """
    Convert a surface mesh into voxel coordinates (interior, surface, or both).
    
    Parameters
    ----------
    mesh : pyvista.DataSet
        The surface mesh.
    spacing : float or (3,) tuple, optional
        Absolute voxel spacing in mesh coordinates.
    origin : (3,) tuple, optional
        Force a specific origin for lattice alignment. If None, uses mesh minimum bounds.
    size : (3,) tuple, optional
        Size of the voxel grid in mesh coordinates. If None, uses mesh bounds.
    ignore_clip : bool, optional
        If True, does not check if the mesh fits completely within the voxel grid. Default is False.
    fill : bool, optional
        If True, fill interior voxels. If False, only surface voxels. Default is True.
    edge_mode : {'center', 'inner', 'outer'}, optional
        'center' selects voxels whose centers are within the mesh surface,
        'inner' selects voxels whose centers are inside or touching the surface from inside,
        'outer' selects voxels whose centers are outside or touching the surface from outside.
    method : {'mesh_to_sdf', 'implicit_distance'}, optional
        Method to compute signed distance field. 'mesh_to_sdf' is more robust for complex meshes. 
        'implicit_distance' uses PyVista's built-in method (often faster).
    
    Returns
    -------
    voxel_volume : (Z, Y, X) ndarray
        3D binary mask indicating voxel occupancy.
    """
    # Ensure triangular mesh TODO: is this necessary?
    mesh = pv.PolyData(mesh).triangulate()
    mins, maxs = mesh.bounds[::2], mesh.bounds[1::2]
    origin_mesh = np.array(mins, dtype=float)
    if origin is not None:
        for i in range(3):
            if origin[i] is not None:
                origin_mesh[i] = origin[i]
    size_mesh = np.array(maxs, dtype=float) - origin_mesh
    if size is not None:
        for i in range(3):
            if size[i] is not None:
                size_mesh[i] = size[i]
        
    if not ignore_clip:
        if np.any(mins < origin_mesh):
            raise ValueError(f"origin {origin_mesh} must be <= mesh minimum bounds {mins}. Use ignore_clip=True to override.")
        if np.any(maxs > origin_mesh + size_mesh):
            raise ValueError(f"origin + size {origin_mesh + size_mesh} must be >= mesh maximum bounds {maxs}. Use ignore_clip=True to override.")
        
    spacing = np.array((spacing, spacing, spacing), float) if np.isscalar(spacing) else np.array(spacing, float)
    nxyz = np.ceil(size_mesh / spacing).astype(int) + 1  # include endpoint
    grid = pv.ImageData(
        dimensions=nxyz,
        spacing=spacing,
        origin=origin_mesh,
    )
    if method == "mesh_to_sdf":
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape((-1, 4))[:, 1:4])
        points = grid.points
        sdf_values = mesh_to_sdf.mesh_to_sdf(trimesh_mesh, points)
    elif method == "implicit_distance":
        distancegrid = grid.compute_implicit_distance(mesh)
        sdf_values = distancegrid.point_data["implicit_distance"]
    sdf = sdf_values.reshape(grid.dimensions[::-1])  # (z,y,x)
    voxel_diagonal = np.linalg.norm(spacing)

    epsilon = 0.1
    if edge_mode == "center":
        mask = np.abs(sdf) <= voxel_diagonal * (0.5)
    elif edge_mode == "inner":
        mask = (sdf <= 0.0) & (sdf >= -voxel_diagonal * (0.5 + epsilon))
    elif edge_mode == "outer":
        mask = (sdf >= 0.0) & (sdf <= voxel_diagonal * (0.5 + epsilon))
    else:
        raise ValueError("edge_mode must be one of {'center','inner','outer'}")
    if fill:
        inside = sdf < 0.0
        mask = inside | mask

    return mask.astype(np.uint8)