import numpy as np
import pyvista as pv
import tifffile

def to_mc_bool_volume(arr: np.ndarray, true_value=None) -> np.ndarray:
    """
    Convert raw array in numpy/ImageJ format (z, y, x) = (Up, South, East) to boolean volume in Minecraft coordinates (x, y, z) = (East, Up, South).
    
    Parameters
    ----------
    arr : ndarray
        Input array in numpy/ImageJ format (z, y, x) or (z, y, x, c). Color channels are summed if present.
    true_value : optional
        The value to be considered as True in the boolean volume. If None, any non-zero value is True.

    Returns
    -------
    result : ndarray
        Boolean volume in Minecraft orientation (x, y, z) = (East, Up, South).
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
    Read a multi-page TIFF file (z, y, x) = (Up, South, East) and return a boolean volume in Minecraft coordinates (x, y, z) = (East, Up, South).

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
    Read a NumPy .npy file (z, y, x) = (Up, South, East) and return a boolean volume in Minecraft coordinates (x, y, z) = (East, Up, South).

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
              minimum: tuple = None,
              maximum: tuple = None,
              origin: tuple = None,
              ignore_clip=False,
              fill: bool = True,
              flip_z: bool = True,
              edge_mode: str = "center",
              method: str = "mesh_to_sdf"
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh (x, y, z) = (East, North, Up) and convert to a boolean voxel volume in Minecraft coordinates (x, y, z) = (East, Up, South).

    Parameters
    ----------
    filename : str
        Path to the mesh file.
    spacing : float or (3,) tuple, optional
        Absolute voxel spacing in mesh coordinates.
    minimum : (3,) tuple, optional
        Minimum bounds of the voxel grid in mesh coordinates. If None, uses mesh minimum bounds.
    maximum : (3,) tuple, optional
        Maximum bounds of the voxel grid in mesh coordinates. If None, uses mesh maximum bounds.
    origin : (3,) tuple, optional
        Origin in mesh coordinates. This is used to calculate the `position` output. If None, uses `minimum`. Default is None.
    ignore_clip : bool, optional
        If True, does not check if the mesh fits completely within the voxel grid. Default is False.
    fill : bool, optional
        If True, fill interior voxels. If False, only surface voxels. Default is True.
    flip_z : bool, optional
        If True, flip the z-axis in the output volume to match Minecraft conventions such that (x, y, z) = (East, Up, South). Default is True.
    edge_mode : {'center', 'inner', 'outer'}, optional
        'center' selects voxels whose centers are within the mesh surface,
        'inner' selects voxels whose centers are inside or touching the surface from inside,
        'outer' selects voxels whose centers are outside or touching the surface from outside.
    method : {'mesh_to_sdf', 'implicit_distance'}, optional
        Method to compute signed distance field. 'mesh_to_sdf' is more robust for complex meshes. 'implicit_distance' uses PyVista's built-in method.
    
    Returns
    -------
    voxel_volume : (x, y, z) ndarray
        3D boolean mask indicating voxel occupancy in Minecraft coordinates.
    position : (3,) ndarray
        Position of the voxel grid origin (zero index) in Minecraft coordinates relative to `origin`.
    """
    vol, position = voxelize_mesh(
        filename,
        spacing=spacing,
        minimum=minimum,
        maximum=maximum,
        origin=origin,
        ignore_clip=ignore_clip,
        fill=fill,
        flip_y=flip_z,
        edge_mode=edge_mode,
        method=method,
    )
    voxel_volume = to_mc_bool_volume(vol)
    position = position[[0, 2, 1]]  # to (x, y, z)
    return voxel_volume, position

def voxelize_mesh(mesh: pv.PolyData | str,
                   spacing: float | tuple = 1.0,
                   minimum: tuple = None,
                   maximum: tuple = None,
                   origin: tuple = None,
                   ignore_clip=False,
                   fill: bool = True,
                   flip_y: bool = True,
                   edge_mode: str = "center",
                   method: str = "mesh_to_sdf"
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh (x, y, z) = (East, North, Up) and convert to a boolean voxel volume (z, y, x) = (Up, South, East).
    
    Parameters
    ----------
    mesh : pyvista.PolyData or str
        The surface mesh or path to the mesh file.
    spacing : float or (3,) tuple, optional
        Absolute voxel spacing in mesh coordinates.
    minimum : (3,) tuple, optional
        Minimum bounds of the voxel grid in mesh coordinates. If None, uses mesh minimum bounds.
    maximum : (3,) tuple, optional
        Maximum bounds of the voxel grid in mesh coordinates. If None, uses mesh maximum bounds.
    origin : (3,) tuple, optional
        Origin in mesh coordinates. This is used to calculate the `position` output. If None, uses `minimum`. Default is None.
    ignore_clip : bool, optional
        If True, does not check if the mesh fits completely within the voxel grid. Default is False.
    fill : bool, optional
        If True, fill interior voxels. If False, only surface voxels. Default is True.
    flip_y : bool, optional
        If True, flip the y-axis in the output volume to match numpy conventions such that (z, y, x) = (Up, South, East). Default is True.
    edge_mode : {'center', 'inner', 'outer'}, optional
        'center' selects voxels whose centers are within the mesh surface,
        'inner' selects voxels whose centers are inside or touching the surface from inside,
        'outer' selects voxels whose centers are outside or touching the surface from outside.
    method : {'mesh_to_sdf', 'implicit_distance'}, optional
        Method to compute signed distance field. 'mesh_to_sdf' is more robust for complex meshes. 
        'implicit_distance' uses PyVista's built-in method (often faster).
    
    Returns
    -------
    voxel_volume : (z, y, x) ndarray
        3D boolean mask indicating voxel occupancy.
    position : (3,) ndarray
        Position of the voxel grid origin (zero index) in voxel coordinates relative to `origin`.
    """
    # Ensure triangular mesh TODO: is this necessary?
    mesh = pv.PolyData(mesh).triangulate()
    mins, maxs = np.array(mesh.bounds[::2], dtype=float), np.array(mesh.bounds[1::2], dtype=float)
    min_mesh = mins.copy()
    if minimum is not None:
        for i in range(3):
            if minimum[i] is not None:
                min_mesh[i] = minimum[i]
    max_mesh = maxs.copy()
    if maximum is not None:
        for i in range(3):
            if maximum[i] is not None:
                max_mesh[i] = maximum[i]
        
    if not ignore_clip:
        if np.any(mins < min_mesh):
            raise ValueError(f"minimum {min_mesh} must be <= the minimum bounds of the mesh {mins}. Use ignore_clip=True to override.")
        if np.any(maxs > max_mesh):
            raise ValueError(f"maximum {max_mesh} must be >= the maximum bounds of the mesh {maxs}. Use ignore_clip=True to override.")
        
    spacing = np.array((spacing, spacing, spacing), float) if np.isscalar(spacing) else np.array(spacing, float)
    nxyz = np.ceil((max_mesh - min_mesh) / spacing).astype(int) + 1  # include endpoint
    grid = pv.ImageData(
        dimensions=nxyz,
        spacing=spacing,
        origin=min_mesh,
    )
    if method == "mesh_to_sdf":
        try:
            import trimesh
            import mesh_to_sdf
        except ImportError:
            raise ImportError("trimesh and mesh_to_sdf are required for 'mesh_to_sdf' method. Please install them via 'pip install trimesh mesh_to_sdf'.")
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
    position = np.zeros(3)
    if origin is not None:
        vox_min = np.ceil(min_mesh / spacing).astype(int)
        vox_origin = np.ceil(np.array(origin, dtype=float) / spacing).astype(int)
        position = vox_min - vox_origin
    if flip_y:
        mask = mask[:, ::-1, :]
        position[1] = - ((mask.shape[1] - 1) + position[1])
    position = position[::-1].astype(int) # to (x, y, z)
    voxel_volume = mask.astype(bool)

    return voxel_volume, position