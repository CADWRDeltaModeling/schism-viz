""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""
import numpy as np
import pandas as pd
import numba
import xarray as xr
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import uxarray as ux


class Grid(ux.Grid):
    """ uxarray Grid class for SCHISM
    """
    elem_polygons = None
    elem_strtree = None
    node_points = None
    node_strtree = None

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

    def build_spatial_trees(self):
        """ Build spatial tree for the nodes and the elements of the grid
        """
        self.elem_strtree = self.build_elem_spatial_tree()
        self.node_strtree = self.build_node_spatial_tree()

    def build_elem_spatial_tree(self):
        """ Build spatial tree for the elements of the grid """
        # If the spatial tree is not built yet, build it
        if self.elem_polygons is None:
            node_x = self.ds[self.ds_var_names['Mesh2_node_x']].values
            node_y = self.ds[self.ds_var_names['Mesh2_node_y']].values

            def create_polygon(node_indices):
                # The node indices are 1-based
                ind = node_indices[node_indices > 0] - 1
                # Assuming the indices are positional
                return Polygon(zip(node_x[ind], node_y[ind]))
            self.elem_polygons = xr.apply_ufunc(create_polygon,
                self.ds[self.ds_var_names['Mesh2_face_nodes']],
                input_core_dims=((self.ds_var_names['nMaxMesh2_face_nodes'],),),
                vectorize=True)
            self.elem_strtree = STRtree(self.elem_polygons.values)
        return self.elem_strtree

    def build_node_spatial_tree(self):
        node_x = self.ds[self.ds_var_names['Mesh2_node_x']].values
        node_y = self.ds[self.ds_var_names['Mesh2_node_y']].values

        def create_point(node_index):
            ind = node_index - 1
            return Point(node_x[ind], node_y[ind])
        self.node_points = [create_point(i) for i in range(
            self.ds.dims[self.ds_var_names['nMesh2_node']])]
        self.node_strtree = STRtree(self.node_points)
        return self.node_strtree

    def find_element_at(self, x, y, predicate='intersects'):
        """ Find the element that contains the point (x, y)

        Parameters
        ----------
        x : float, required
            x coordinate
        y : float, required
            y coordinate
        predicate : str, optional
            Predicate to use for the spatial query by shapely STRtree.
            Default is 'contains'

        Returns
        -------
        elem : array_like
            Element indices
        """
        if self.elem_strtree is None:
            self.build_elem_spatial_tree()
        point = Point(x, y)
        return self.elem_strtree.query(point, predicate=predicate)


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
    ds['SCHISM_hgrid_face_nodes'] = xr.DataArray(data=df_faces[[2, 3, 4, 5]].astype(int).values,
                                                 dims=("nSCHISM_hgrid_face",
                                                       "nMaxSCHISM_hgrid_face_nodes"),
                                                 attrs={"_FillValue": -1})
    # Add dummy mesh_topology variable
    ds = Grid.add_topology_variable(ds)

    grid = Grid(ds, islation=False, mesh_type="ugrid")
    return grid
