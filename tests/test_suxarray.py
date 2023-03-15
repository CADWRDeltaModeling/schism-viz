from pathlib import Path
import numpy as np
import xarray as xr
import pytest
import schismviz.suxarray as sx


def test_suxarray_init_with_out2d():
    """ Test suxarray initialization with a SCHISM out2d file """
    # Test with a HelloSCHISM v5.10 out2d file
    p_cur = Path(__file__).parent.absolute()
    ds = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"))
    grid = sx.Grid(ds)
    assert grid.mesh_type == 'ugrid'
    assert grid.ds.dims['nSCHISM_hgrid_node'] == 2639


def test_get_topology_variable():
    """ Test get_topology_variable """
    # Test with a HelloSCHISM v5.10 out2d file
    p_cur = Path(__file__).parent.absolute()
    ds = xr.open_dataset(str(p_cur / "testdata/out2d_1.nc"))
    da = sx.get_topology_variable(ds)
    assert da is None
    # Add a dummy topology variable
    ds = sx.Grid.add_topology_variable(ds)
    da = sx.get_topology_variable(ds)
    assert da.name == 'SCHISM'


@pytest.fixture
def test_grid():
    """ Test mesh fixture """
    p_cur = Path(__file__).parent.absolute()
    grid = sx.read_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    return grid


def test_triangulate(test_grid):
    """ Test triangulate """
    grid_tri = sx.triangulate(test_grid)
    assert grid_tri.ds.dims['nSCHISM_hgrid_node'] == 112
    assert grid_tri.ds.dims['nSCHISM_hgrid_face'] == 168


def test_read_hgrid_gr3():
    """ Test read_hgrid_gr3 """
    # Test with a HelloSCHISM v5.10 hgrid.gr3 file
    p_cur = Path(__file__).parent.absolute()
    grid = sx.read_hgrid_gr3(str(p_cur / "testdata/testmesh.gr3"))
    assert grid.mesh_type == 'ugrid'
    assert grid.ds.dims['nSCHISM_hgrid_node'] == 112
    assert grid.ds.dims['nSCHISM_hgrid_face'] == 135
    # assert grid.ds.dims['nSCHISM_hgrid_edge'] == 10416
    # assert grid.ds.dims['nSCHISM_hgrid_max_face_nodes'] == 3
    # assert grid.ds.dims['nSCHISM_hgrid_max_edge_nodes'] == 2


def test_find_element_at_position(test_grid):
    """ Test find_element_at """
    # When a point is inside an element
    elem_ind = test_grid.find_element_at(2., 1.)
    assert np.all(elem_ind == np.array([123]))
    # When a point is on a boundary of two elements
    elem_ind = test_grid.find_element_at(0., 0.)
    assert np.all(elem_ind == np.array([39, 123]))
