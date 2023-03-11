""" suxarray module

`suxarray` is a module that extends the functionality of `uxarray` for the
SCHISM grid.
"""

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
        if len(dataset.filter_by_attrs(cf_role="mesh_topology").keys()) == 0:
            dataset = self.add_SCHISM_mesh_topology_dummy_variable(dataset)
        # Initialize the super class
        super().__init__(dataset, **kwargs)

    @staticmethod
    def add_SCHISM_mesh_topology_dummy_variable(ds, varname="Mesh2"):
        ds = ds.assign({varname: 1})
        ds[varname].attrs['cf_role'] = 'mesh_topology'
        ds[varname].attrs['topology_dimension'] = 2
        ds[varname].attrs['node_coordinates'] = "SCHISM_hgrid_node_x SCHISM_hgrid_node_y"
        ds[varname].attrs['face_node_connectivity'] = "SCHISM_hgrid_face_nodes"
        ds[varname].attrs['start_index'] = 1
        ds[varname].attrs['_FillValue'] = -1
        return ds
