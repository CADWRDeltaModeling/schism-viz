import xarray as xr
import pandas as pd
from schimpy import schism_mesh
from .trimesh_animator import ColorValueAnimator

def build_color_value_animator(mesh_file, value_file, color_value='salinity'):
    '''
    mesh file for SCHISM : e.g. hgrid.gr3
    value file for SCHISM: e.g. salinity.nc
    '''
    # load data from files and convert to map crs
    mesh = schism_mesh.read_mesh(mesh_file) # read mesh file
    dfvalues = xr.open_dataset(value_file) # xarray but try dataframe
    dfvertices = pd.DataFrame(mesh.elems) # vertices (3 columns with id to the mesh.nodes)
    dfnodes=pd.DataFrame(mesh.nodes,columns=['x','y','z']) # mesh nodes have x,y,z spatial position
    dfnodes[color_value] = dfvalues[color_value].values[0,:,0] #  assumption that value slices are time (1st), color_value (2nd), depth (3rd)
    # make animator
    return ColorValueAnimator(dfvertices, dfnodes, dfvalues)
