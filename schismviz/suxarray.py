""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""
import numpy as np
import pandas as pd
import numba
import xarray as xr
import uxarray as ux


class Grid(ux.Grid):
    """ uxarray Grid class for SCHISM
    """

    def __init__(self, dataset, **kwargs):
        """ Initialize a Grid object

        Parameters
        ----------
        dataset : xarray.Dataset, ndarray, list, tuple, required
            Input xarray.Dataset or vertex coordinates that form one face.

        Other Parameters
        ----------------
        islatlon : bool, optional
            Specify if the grid is lat/lon based
        concave: bool, optional
            Specify if this grid has concave elements (internal checks for this are possible)
        gridspec: bool, optional
            Specifies gridspec
        mesh_type: str, optional
            Specify the mesh file type, eg. exo, ugrid, shp, etc
        """
        # Add a topology dummy variable if it doesn't exist
        # The current SCHISM out2d does not have this variable.
        if get_topology_variable(dataset) is None:
            dataset = self.add_topology_variable(dataset)
        # Initialize the super class
        super().__init__(dataset, **kwargs)

    @staticmethod
    def add_topology_variable(ds, varname="SCHISM"):
        """ Add a dummy mesh_topology variable to a SCHISM out2d dataset

        Parameters
        ----------
        ds : xarray.Dataset, required
            Input SCHISM out2d xarray.Dataset
        varname : str, optional
            Name of the dummy topology variable. Default is "Mesh2"
        """
        ds = ds.assign({varname: 1})
        ds[varname].attrs['cf_role'] = 'mesh_topology'
        ds[varname].attrs['topology_dimension'] = 2
        ds[varname].attrs['node_coordinates'] = "SCHISM_hgrid_node_x SCHISM_hgrid_node_y"
        ds[varname].attrs['face_node_connectivity'] = "SCHISM_hgrid_face_nodes"
        ds[varname].attrs['start_index'] = 1
        ds[varname].attrs['_FillValue'] = -1
        return ds


def get_topology_variable(dataset):
    """ Get the topology xarray.DataArray

    Parameters
    ----------
    dataset : xarray.Dataset, required
        Input xarray.Dataset

    Returns
    -------
    da : xarray.DataArray
        Topology Xarray.DataArray
    """
    ds = dataset.filter_by_attrs(cf_role="mesh_topology")
    if len(ds) == 0:
        return None
    elif len(ds) > 1:
        raise ValueError("Multiple mesh_topology variables found")
    else:
        return ds[list(ds.keys())[0]]


def triangulate(grid):
    """ Triangulate a suxarray grid

    Parameters
    ----------
    grid : Grid, required
        Grid object to triangulate

    Returns
    -------
    grid : Grid
        Triangulated grid
    """
    mesh_name = grid.ds_var_names['Mesh2']
    face_nodes = grid.ds[grid.ds_var_names['Mesh2_face_nodes']]
    n_face, _ = face_nodes.shape
    fill_value = grid.ds[mesh_name].attrs["_FillValue"]
    valid = face_nodes != fill_value
    n_per_row = valid.sum(axis=1)
    n_triangle_per_row = n_per_row - 2
    face_ori = np.repeat(np.arange(n_face), n_per_row)
    node_ori = face_nodes.values.ravel()[valid.values.ravel()]

    def _triangulate(face_ori: np.ndarray, node_ori: np.ndarray,
                     n_triangle_per_row: xr.DataArray) -> np.ndarray:
        n_triangle = n_triangle_per_row.sum().item()
        n_face = len(face_ori)
        index_first = np.argwhere(np.diff(face_ori, prepend=-1) != 0)
        index_second = index_first + 1
        index_last = np.argwhere(np.diff(face_ori, append=-1) != 0)

        first = np.full(n_face, False)
        first[index_first] = True
        second = np.full(n_face, True) & ~first
        second[index_last] = False
        third = np.full(n_face, True) & ~first
        third[index_second] = False

        triangles = np.empty((n_triangle, 3), np.int32)
        triangles[:, 0] = np.repeat(node_ori[first], n_triangle_per_row)
        triangles[:, 1] = node_ori[second]
        triangles[:, 2] = node_ori[third]
        return triangles

    triangles = _triangulate(face_ori, node_ori, n_triangle_per_row)

    triangle_original_ind = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row)

    # Copy the data from the original grid
    ds_tri = grid.ds.copy()
    # Drop the original face_nodes variable
    ds_tri = ds_tri.drop_vars(grid.ds_var_names['Mesh2_face_nodes'])
    da_face_nodes = xr.DataArray(data=triangles,
                           dims=(f"n{mesh_name}_hgrid_face", "three"),
                           name=f"{mesh_name}_hgrid_face_nodes")
    ds_tri[da_face_nodes.name] = da_face_nodes
    da_elem_ind = xr.DataArray(data=triangle_original_ind,
                               dims=(f"n{mesh_name}_hgrid_face"),
                               name=f"{mesh_name}_face_original")
    ds_tri[da_elem_ind.name] = da_elem_ind
    grid_tri = ux.Grid(ds_tri, islation=False, mesh_type="ugrid")
    # grid_tri.Mesh2.attrs['start_index'] = 0
    return grid_tri


def read_hgrid_gr3(path_hgrid):
    """ Read SCHISM hgrid.gr3 file and return suxarray grid """
    # read the header
    with open(path_hgrid, "r") as f:
        first_line = f.readline()
        n_faces, n_nodes = [int(x) for x in f.readline().strip().split()[:2]]
    # Read the node section. Read only up to the fourth column
    df_nodes = pd.read_csv(path_hgrid, skiprows=2, header=None,
                           nrows=n_nodes, sep="\s+", usecols=range(4))
    # Read the face section. Read only up to the sixth column. The last column
    # may exist or not.
    df_faces = pd.read_csv(path_hgrid, skiprows=2 + n_nodes, header=None,
                           nrows=n_faces, sep="\s+", names=range(6))
    # TODO Read boundary information, if any
    # Create suxarray grid
    ds = xr.Dataset()
    ds['SCHISM_hgrid_node_x'] = xr.DataArray(data=df_nodes[1].values, dims="nSCHISM_hgrid_node")
    ds['SCHISM_hgrid_node_y'] = xr.DataArray(data=df_nodes[2].values, dims="nSCHISM_hgrid_node")
    # Replace NaN with -1
    df_faces = df_faces.fillna(-1)
    ds['SCHISM_hgrid_face_nodes'] = xr.DataArray(data=df_faces[[2, 3, 4, 5]].values,
                                                 dims=("nSCHISM_hgrid_face", "nMaxSCHISM_hgrid_face_nodes"))
    # Add dummy mesh_topology variable
    ds = Grid.add_topology_variable(ds)

    grid = Grid(ds, islation=False, mesh_type="ugrid")
    return grid
