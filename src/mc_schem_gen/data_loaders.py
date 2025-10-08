import numpy as np
import pyvista as pv
import vtk
import tifffile

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
              edge_mode: str = "center"
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
        ),
    )

def voxelize_mesh(mesh: pv.DataSet,
                   spacing: float = 1.0,
                   origin: tuple = None,
                   size: tuple = None,
                   ignore_clip=False,
                   fill: bool = True,
                   edge_mode: str = "center") -> np.ndarray:
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
    
    Returns
    -------
    voxel_volume : (Z, Y, X) ndarray
        3D binary mask indicating voxel occupancy.
    """
    # Ensure triangular mesh TODO: is this necessary?
    mesh = pv.PolyData(mesh).triangulate()
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    if origin is None:
        origin = (xmin, ymin, zmin)
    origin = np.array(origin, dtype=float)
    size_mesh = np.array([xmax - origin[0], ymax - origin[1], zmax - origin[2]], dtype=float)
    if size is not None:
        for i in range(3):
            if size[i] is not None:
                size_mesh[i] = size[i]
        
    if not ignore_clip:
        if origin[0] > xmin or origin[1] > ymin or origin[2] > zmin:
            raise ValueError(f"origin {origin} must be <= mesh minimum bounds {(xmin, ymin, zmin)}")
        if origin[0] + size_mesh[0] > xmax or origin[1] + size_mesh[1] > ymax or origin[2] + size_mesh[2] > zmax:
            raise ValueError(f"origin + size {origin + size_mesh} must be >= mesh maximum bounds {(xmax, ymax, zmax)}")
        
    spacing = np.array((spacing, spacing, spacing), float) if np.isscalar(spacing) else np.array(spacing, float)
    nxyz = np.ceil(size_mesh / spacing).astype(int) + 1  # include endpoint
    grid = pv.ImageData(
        dimensions=nxyz,
        spacing=spacing,
        origin=origin,
    )
    pts = grid.points
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)
    # Accurate but slow per-point evaluation. TODO can this be optimized?
    distances = np.array([imp.EvaluateFunction(p) for p in pts], dtype=np.float32)
    sdf = distances.reshape(grid.dimensions[::-1])  # (z,y,x)
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

    # flip the y axis to match image coordinates #TODO is this necessary?
    mask = np.flip(mask, axis=1)
    return mask.astype(np.uint8)