# Codes to plot SCHISM binary outputs using HoloViews.
#
# First-class data structures are xarray/uxarray. The codes avoid using SCHISM
# specific data structures as much as possible.

import datetime
import numpy as np
import xarray as xr
import uxarray as ux
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import holoviews as hv
from holoviews.operation.datashader import rasterize
from typing import Tuple

__all__ = ['create_mesh_plot', 'create_plot_with_point_select']


def plot_mesh(ds_out2d: xr.Dataset, dataarray: xr.DataArray):
    """ Create a dynamic HoloViews 2D TriMesh plot using a DataArray value

    Plot a HoloViews TrimMesh colored with values from the DataArray using
    the mesh information from ds_out2d.
    The function assumes that the dimensions and the coordinates from ds_out2d
    and dataarray agree.

    NOTE: This function supports only 2D value DataArray at the moment.

    Parameters
    ----------
    ds_out2d: xarray.Dataset
        Dataset containing the mesh information. Use any one of out2d files from
        SCHISM outputs.
    dataarray: xarray.DataArray
        DataArray containing the values to be plotted.

    Returns
    -------
    HoloViews plot

    Examples
    --------
    >>> p = plot_mesh(ds_out2d, ds_out2d.elevation)
    """
    meshobj = HvTriMesh(ds_out2d)
    plot = meshobj.create_plot(dataarray)

    return plot


def plot_mesh_with_point_select(ds_out2d, dataarray):
    """ Create a dynamic HoloViews 2D TriMesh plot with a point selector to plot
    a time series plot at the nearest node from the selected point.

    Plot a HoloViews TrimMesh colored with values from the DataArray using
    the mesh information from ds_out2d.
    The function assumes that the dimensions and the coordinates from ds_out2d
    and dataarray agree.

    NOTE: This function supports only 2D value DataArray at the moment.

    Parameters
    ----------
    ds_out2d: xarray.Dataset
        Dataset containing the mesh information. Use any one of out2d files from
        SCHISM outputs.
    dataarray: xarray.DataArray
        DataArray containing the values to be plotted.

    Returns
    -------
    HoloViews plot

    Examples
    --------
    >>> p = plot_mesh(ds_out2d, ds_out2d.elevation)
    """
    meshobj = HvTriMeshWithPointer(ds_out2d)
    plot = meshobj.create_plot(dataarray)

    return plot


class HvTriMesh:
    """ Base class for the TriMesh plotter
    """

    def __init__(self, ds_out2d: xr.Dataset):
        """ Initialize the TriMesh plotter

        Parameters
        ----------
        ds_out2d: xarray.Dataset
            Dataset with the mesh information
        """
        self._ds_out2d = ds_out2d
        self._process_mesh()

    def _process_mesh(self):
        self.ux_grid = triangulate_uxgrid(make_uxgrid(self._ds_out2d))
        self.node_x = self.ux_grid.Mesh2_node_x.values
        self.node_y = self.ux_grid.Mesh2_node_y.values
        self.node_z = self.ux_grid.ds.depth.values
        self.node_coordinates = np.column_stack((self.node_x, self.node_y, self.node_z))
        self.nodes = hv.Points(self.node_coordinates, vdims="z")

    def create_timestamps(self, da: xr.DataArray):
        base_date = [int(float(x)) for x in
                     da.time.attrs['base_date'].strip().split()]
        time_basis = datetime.datetime(*base_date[:4])
        times = [time_basis + datetime.timedelta(seconds=t) for t in da.time.values]
        return times

    def dynamic_mesh(self, dataarray, index):
        self.nodes.data[:, -1] = dataarray.sel(time=index).values
        mesh = hv.TriMesh((self.ux_grid.ds.Mesh2_face_nodes.values, self.nodes))
        return mesh

    def prepare_plot(self, dataarray: xr.DataArray):
        times = self.create_timestamps(dataarray)
        self.dataarray = dataarray.assign_coords(time=times)
        self.dim_time = hv.Dimension('time', values=times)

    def create_plot(self, dataarray: xr.DataArray):
        """ Create a TriMesh plot
        """
        self.prepare_plot(dataarray)
        plot = rasterize(hv.DynamicMap(lambda i: self.dynamic_mesh(self.dataarray, i),
                                       kdims=self.dim_time))

        return plot


class HvTriMeshWithPointer(HvTriMesh):
    def __init__(self, ds_out2d: xr.Dataset):
        super().__init__(ds_out2d)

    def _process_mesh(self):
        super()._process_mesh()
        self.elem_polygons = None
        self.node_points = None
        self.build_strtree_of_elements()
        self.build_strtree_of_points()

    def build_strtree_of_elements(self):
        if self.elem_polygons is None:
            self.elem_polygons = xr.apply_ufunc(lambda x: Polygon(self.node_coordinates[x, :2]),
                self.ux_grid.ds.Mesh2_face_nodes,
                input_core_dims=(('three',),), vectorize=True)
            self.elem_strtree = STRtree(self.elem_polygons.values)
        return self.elem_strtree

    def build_strtree_of_points(self):
        if self.node_points is None:
            self.node_points = [Point(x) for x in self.node_coordinates[:, :2]]
            self.node_strtree = STRtree(self.node_points)
        return self.node_strtree

    def create_select_points(self):
        selected_points = hv.Points(([], [], []), vdims='z')
        point_stream = hv.streams.PointDraw(
            data=selected_points.columns(), num_objects=1, source=selected_points)
        return selected_points, point_stream

    def find_nearest_node(self, elem_i, pos):
        nodes_i = self.ux_grid.ds.Mesh2_face_nodes[elem_i, :].values[0]
        pts = [Point(self.node_coordinates[n_i, :2]) for n_i in nodes_i]
        p_pos = Point(pos)
        dists = [p.distance(p_pos) for p in pts]
        return nodes_i[np.argmin(dists)]

    def dynamic_curve(self, dataarray, data):
        if len(data['x']) == 0:
            # Need to provide a dummy x values for datetime
            p = hv.Curve((dataarray.time.isel(time=0), [0.]))
            return p
        pos = (data['x'][0], data['y'][0])
        elem_i = self.elem_strtree.query(Point(pos), predicate='intersects')
        node_i = self.find_nearest_node(elem_i, pos)
        p = hv.Curve((dataarray.time, dataarray.sel(nSCHISM_hgrid_node=node_i)))
        return p

    def create_plot(self, dataarray: xr.DataArray):
        self.prepare_plot(dataarray)
        self.selected_points, self.point_stream = self.create_select_points()
        mesh = rasterize(hv.DynamicMap(lambda i: self.dynamic_mesh(self.dataarray, i),
                                       kdims=self.dim_time))
        curve = hv.DynamicMap(lambda data:
                              self.dynamic_curve(self.dataarray, data),
                              streams=[self.point_stream]).opts(framewise=True)
        mesh_with_pointer = (mesh * self.selected_points).opts(
            hv.opts.Layout(merge_tools=False),
            hv.opts.Points(active_tools=['point_draw']))

        return (mesh_with_pointer + curve).opts(shared_axes=False).cols(1)


def triangulate(face_node_connectivity: np.ndarray,
                fill_value: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Create new triangle connectivity from hybrid mesh face-node connectivity

    Create a new connectivity array for triangles split from the original mesh.
    A SCHISM mesh is a hybrid mesh that contains both triangles and quadrilaterals.
    So, the elements need to be triangulated before plotting with HoloViews
    TriMesh.

    This code is largely copied-and-pasted from uxgrid.

    Parameters
    ----------
    face_node_connectivity: np.ndarray
        Face node connectivity
    fill_value: int
        Fill value

    Returns
    -------
    triangles: np.ndarray
        New triangle connectivity
    original_elem_indices : np.ndarray
        The original element indices for each triangle
    """
    n_face, n_max = face_node_connectivity.shape

    valid = face_node_connectivity != fill_value
    n_per_row = valid.sum(axis=1)
    n_triangle_per_row = n_per_row - 2
    i = np.repeat(np.arange(n_face), n_per_row)
    j = face_node_connectivity.ravel()[valid.ravel()]

    def _triangulate(i: np.ndarray, j: np.ndarray,
                     n_triangle_per_row: np.ndarray) -> np.ndarray:
        n_triangle = n_triangle_per_row.sum()
        n_face = len(i)
        index_first = np.argwhere(np.diff(i, prepend=-1) != 0)
        index_second = index_first + 1
        index_last = np.argwhere(np.diff(i, append=-1) != 0)

        first = np.full(n_face, False)
        first[index_first] = True
        second = np.full(n_face, True) & ~first
        second[index_last] = False
        third = np.full(n_face, True) & ~first
        third[index_second] = False

        triangles = np.empty((n_triangle, 3), np.intp)
        triangles[:, 0] = np.repeat(j[first], n_triangle_per_row)
        triangles[:, 1] = j[second]
        triangles[:, 2] = j[third]
        return triangles

    triangles = _triangulate(i, j, n_triangle_per_row)

    original_elem_indices = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row
    )
    return triangles, original_elem_indices


def triangulate_uxgrid(grid: ux.grid.Grid):
    """ Create a new uxarray Grid with triangles only

    Create a new uxarray grid data set for triangles split from the original mesh.

    This code is largely copied-and-pasted from uxgrid.

    Parameters
    ----------
    grid: uxarray.grid.Grid
        The original grid

    Returns
    -------
    grid: uxarray.grid.Grid
        A triangulated grid
    """
    n_face, n_max = grid.Mesh2_face_nodes.shape

    face_node_connectivity = grid.Mesh2_face_nodes

    fill_value = grid.Mesh2._FillValue
    valid = grid.Mesh2_face_nodes != fill_value
    n_per_row = valid.sum(axis=1)
    n_triangle_per_row = n_per_row - 2
    i = np.repeat(np.arange(n_face), n_per_row)
    j = face_node_connectivity.values.ravel()[valid.values.ravel()]

    def _triangulate(i: np.ndarray, j: np.ndarray, n_triangle_per_row: np.ndarray) -> np.ndarray:
        n_triangle = n_triangle_per_row.sum().values.item()
        n_face = len(i)
        index_first = np.argwhere(np.diff(i, prepend=-1) != 0)
        index_second = index_first + 1
        index_last = np.argwhere(np.diff(i, append=-1) != 0)

        first = np.full(n_face, False)
        first[index_first] = True
        second = np.full(n_face, True) & ~first
        second[index_last] = False
        third = np.full(n_face, True) & ~first
        third[index_second] = False

        triangles = np.empty((n_triangle, 3), np.intp)
        triangles[:, 0] = np.repeat(j[first], n_triangle_per_row)
        triangles[:, 1] = j[second]
        triangles[:, 2] = j[third]
        return triangles

    triangles = _triangulate(i, j, n_triangle_per_row)

    triangle_original_ind = np.repeat(
        np.arange(n_face), repeats=n_triangle_per_row
    )

    ds_tri = grid.ds.copy()
    da_conn = xr.DataArray(data=triangles, dims=("nSCHISM_hgrid_tri_face", "three"),
                           name="Mesh2_face_nodes") - 1
    da_elem_ind = xr.DataArray(data=triangle_original_ind, dims=("nSCHISM_hgrid_tri_face"),
                           name="Mesh2_original_face")
    ds_tri.update({"Mesh2_face_nodes": da_conn,
                   "Mesh2_original_face": da_elem_ind})
    grid_tri = ux.Grid(ds_tri, islation=False, mesh_type="ugrid")
    grid_tri.Mesh2.attrs['start_index'] = 0
    return grid_tri


def make_uxgrid(ds: xr.Dataset) -> ux.Grid:
    """ Make a uxarray grid dataset from SCHISM out2d

    Parameters
    ----------
    ds: xarray.Dataset
        SCHISM out2d

    Returns
    -------
    grid2d: uxarray.Grid
    """
    ds = ds.copy()
    ds = ds.assign(Mesh2=1)
    ds.Mesh2.attrs['cf_role'] = 'mesh_topology'
    ds.Mesh2.attrs['topology_dimension'] = 2
    ds.Mesh2.attrs['node_coordinates'] = "SCHISM_hgrid_node_x SCHISM_hgrid_node_y"
    ds.Mesh2.attrs['face_node_connectivity'] = "SCHISM_hgrid_face_nodes"
    ds.Mesh2.attrs['start_index'] = 1
    ds.Mesh2.attrs['_FillValue'] = -1
    grid2d = ux.Grid(ds, islation=False, mesh_type="ugrid")
    return grid2d
