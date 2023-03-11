from pathlib import Path
import xarray as xr
import schismviz.suxarray as sx


def test_suxarray_init_with_out2d():
    """ Test suxarray initialization with a SCHISM out2d file """
    # Test with a HelloSCHISM v5.10 out2d file
    p_cur = Path(__file__).parent.absolute()
    ds = xr.open_dataset(str(p_cur / "data/out2d_1.nc"))
    grid = sx.Grid(ds)
    assert grid.mesh_type == 'ugrid'
    assert grid.ds.dims['nSCHISM_hgrid_node'] == 2639
