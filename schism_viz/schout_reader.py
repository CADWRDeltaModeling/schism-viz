# -*- coding: utf-8 -*-
# Python 3
# PoC to read information from uncombined schout NetCDF files
import os
import glob
import re
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
from schimpy import schism_mesh

__all__ = ['read_schout']


class SchoutUncombinedMesh:
    def __init__(self, *args, path_outputs=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        path_outputs: str, optional
            The path to the output directory
        """
        self._schism_mesh = None
        self._path_outputs = path_outputs if path_outputs is not None else './outputs'
        self.read_global_geometry(self._path_outputs)

    def _allocate_memory(self):
        self._nodes = np.empty((self.global_dims['np'], 3), dtype=np.float64)
        self._kbp = np.empty((self.global_dims['np'], ), dtype=np.int32)
        self._elems = np.empty((self.global_dims['ne'], 4), dtype=np.int32)

    def _create_geometry_variables(self):
        self.elems_local_to_global = []
        self.nodes_local_to_global = []
        self.sides_local_to_global = []
        self.nodes_local = []
        self.elems_local = []

    def read_global_geometry(self, path_outputs=None):
        """
        Gather global geometry from uncombined nc outputs

        Parameters
        ----------
        path_outputs: str, optional
            the path of the outputs where local_to_global files can be found
        """
        if path_outputs is None:
            path_outputs = self._path_outputs
        fname = os.path.join(path_outputs, 'local_to_global_0000')
        self.global_dims = self.read_global_dimension(fname)
        self._allocate_memory()
        self._create_geometry_variables()
        self.proc_header = []
        for rank in range(self.global_dims['nproc']):
            fname = os.path.join(
                path_outputs, 'local_to_global_{:04d}'.format(rank))
            with open(fname, 'r') as fobj:
                fobj.readline()
                fobj.readline()
                self.elems_local_to_global.append(
                    self.read_elements_local_to_global(fobj))
                self.nodes_local_to_global.append(
                    self.read_nodes_local_to_global(fobj))
                self.sides_local_to_global.append(
                    self.read_sides_local_to_global(fobj))
                header = self.read_header_local_to_global(fobj)
                self.proc_header.append(header)
                self.read_nodes(fobj, self.n_nodes_local)
                self.read_elems(fobj, self.n_elems_local)

    @property
    def schism_mesh(self):
        if self._schism_mesh is None:
            self._schism_mesh = schism_mesh.SchismMesh()
            self._schism_mesh._nodes = self._nodes
            self._schism_mesh._elems = self._elems
        return self._schism_mesh

    def read_global_dimension(self, path_local_to_global):
        """  Read the global dimension from the header of local_to_global

            Returns
            -------
            global dimensions: dict
                [ns, ne, np, nvrt, nproc]
        """
        names = ['ns', 'ne', 'np', 'nvrt', 'nproc']
        with open(path_local_to_global, 'r') as fin:
            return dict(zip(names,
                            [int(x) for x in fin.readline().split()[:5]]))

    def read_elements_local_to_global(self, fobj):
        """ Read the element local to global section from a local_to_global file
        """
        n_elems_local = int(fobj.readline().strip())  # # of local elements
        elems_local_to_global = np.genfromtxt(
            fobj, max_rows=n_elems_local, dtype=np.int32)
        elems_local_to_global -= 1  # to zero-based
        return elems_local_to_global

    def read_nodes_local_to_global(self, fobj):
        """ Read the node local to global section from a local_to_global file
        """
        n_nodes_local = int(fobj.readline().strip())  # # of local nodes
        nodes_local_to_global = np.genfromtxt(
            fobj, max_rows=n_nodes_local, dtype=np.int32)
        nodes_local_to_global -= 1  # to zero-based
        return nodes_local_to_global

    def read_sides_local_to_global(self, fobj):
        """ Read the side local to global section from a local_to_global file
        """
        n_sides_local = int(fobj.readline().strip())  # # of local sides
        sides_local_to_global = np.genfromtxt(
            fobj, max_rows=n_sides_local, dtype=np.int32)
        sides_local_to_global -= 1  # to zero-based
        return sides_local_to_global

    def read_header_local_to_global(self, fobj):
        """ Read the header section of the end of a local_to_global file
        """
        # For now, the lines below are not being read.
        fobj.readline()  # "Header:"
        l = fobj.readline().split()  # start_year,start_month,start_day,start_hour
        start = [int(x) for x in l[:3]]
        l = fobj.readline()  # utc_start
        l = fobj.readline()  # nrec,dtout,nspool,nvrt,kz, h0
        tkns = l.strip().split()
        nrec = int(tkns[0])
        nvrt = int(tkns[3])
        l = fobj.readline()  # h_s,h_c,theta_b,theta_f,itmp
        count = nvrt
        while count > 0:
            l = fobj.readline()  # ztot
            count -= len(l.strip().split())
        l = fobj.readline()  # np_lcl, ne_lcl
        self.n_nodes_local, self.n_elems_local = [
            int(x) for x in l.split()[:2]]
        # if n_nodes != self.nodes_local_to_global[-1].shape[0]:
        #     print(len(self.nodes_local_to_global))
        #     raise ValueError("The number of the local nodes do not agree.")
        # if n_elems != self.elems_local_to_global[-1].shape[0]:
        #     raise ValueError("The number of the local elements do not agree.")
        return {'nrec': nrec}

    def read_nodes(self, fobj, n_nodes):
        """ Read the nodes section from a local_to_global file
        """
        nodes = np.genfromtxt(fobj, max_rows=n_nodes)   # x, y, depth, kbp
        self._nodes[self.nodes_local_to_global[-1][:, 1], :] = nodes[:, :3]
        self._kbp[self.nodes_local_to_global[-1][:, 1]] = nodes[:, 3]

    def read_elems(self, fobj, n_elems):
        for elem_i_local in range(n_elems):
            l = fobj.readline()
            tkn = [self.nodes_local_to_global[-1]
                   [int(x) - 1, 1] for x in l.split()[1:]]
            if len(tkn) == 3:
                # Add int32 min (negative) for a missing value
                tkn.append(np.iinfo(np.int32).min)
            elif len(tkn) != 4:
                raise ValueError(
                    'The number of items in the connectivity is not correct.')
            tkn = np.array(tkn, dtype=np.int32)
            self._elems[self.elems_local_to_global[-1][elem_i_local, 1]] = tkn


class Schout:
    """ Class to hold SCHISM local to global geometry information
    """

    def __init__(self, *args, path_outputs=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        path_outputs: str, optional
            The path to the output directory
        """
        self._path_outputs = path_outputs if path_outputs is not None else './outputs'
        self.mesh = SchoutUncombinedMesh(
            path_outputs=self._path_outputs)

    @property
    def schism_mesh(self):
        return self.mesh.schism_mesh

    def get_available_spools(self):
        """ Get the available spool from the outputs direcotry.
        """
        schout_files = glob.glob(os.path.join(
            self._path_outputs, 'schout_0000_*.nc'))
        spools = [int(list(filter(None, re.split(r"[^\d]", f)))[-1])
                  for f in schout_files]
        return spools

    def get_available_variable_names(self):
        schout_files = glob.glob(os.path.join(
            self._path_outputs, 'schout_0000_*.nc'))
        with nc.Dataset(schout_files[0], 'r') as f:
            return f.variables.keys()

    def point_value(self, variable_name, x, y, z, spool_begin, spool_end):
        pass

    def node_value(self, variable_name, rank, node_local_i, spool_begin, spool_end,
                   t_basis=None):
        """ Extract a time series of a variable at a node from the uncombined
            schout_*.nc files.
            NOTE: It is a work-in-progress, so it assumes that the variable is
            a 2D one. The type of the return value is a simple 1D numpy arary.

            Parameters
            ----------
            variable_name: str
                variable name to extract.
            rank: int
                rank or processor to read (zero-based)
            node_local_i: int
                local node index to retrieve (zero-based)
            spool_begin: int
                spool index to start retrieving (one-based)
            spool_end: int
                spool index to end retrieving (inclusive, one-based)

            Returns
            -------
            numpy.array
                variable
        """
        # TODO: Switch to xarray
        n_spools = spool_end - spool_begin + 1

        xr_datasets = []
        for spool_i in range(spool_begin, spool_end + 1):
            path_schout = path_schout = os.path.join(
                    self._path_outputs, 'schout_{:04d}_{:d}.nc'.format(rank, spool_i))
            xr_schout = xr.open_dataset(path_schout)
            xr_datasets.append(xr_schout[variable_name].sel(nSCHISM_hgrid_node=node_local_i))
        xr_result = xr.concat(xr_datasets, dim='time')

        if t_basis is not None:
            times = [t_basis + pd.to_timedelta(t, unit='s') for t in xr_result['time'].values]
            xr_result['time'] = times
        return xr_result

    def available_variables(self):
        """ Get the list of variable names from the first file among
            schout_0000_*.nc.

            Returns
            -------
            list
                list of variable names
        """
        schout_files = glob.glob(os.path.join(
            self._path_outputs, 'schout_0000_*.nc'))
        path_file = os.path.join(self._path_outputs, schout_files[0])
        with nc.Dataset(path_file, 'r') as root:
            return root.variables.keys()

    def variable_rank(self, name):
        """ Get the rank of the variable from the nc file
        """
        schout_files = glob.glob(os.path.join(
            self._path_outputs, 'schout_0000_*.nc'))
        path_file = os.path.join(self._path_outputs, schout_files[0])
        with nc.Dataset(path_file, 'r') as root:
            var = root.variables[name]
            return len(var.shape)

    def variable(self, name, spool_begin, spool_end, skip=1, node_i=None, level=None):
        """ Read a variable of the whole domain
            from the uncombined SCHISM schout files.

            Parameters
            ----------
            name: str
                variable name to read.
            spool_begin: int
                spool number to start reading.
            spool_end: int
                spool number to end reading (inclusive).
            skip: int, optional
                skip size in the time dimension. default = 1, no skipping.
            level: int, optional
                level number to read in, if it is given.
                If not, all levels will be read in.

            Returns
            -------
            numpy.ndarray
                data for the variable
        """
        if node_i is not None:
            raise NotImplementedError()
        nproc = self.mesh.global_dims['nproc']
        n_spools = spool_end - spool_begin + 1
        nrec = self.mesh.proc_header[0]['nrec']
        i23d = -1
        ivs = -1
        schout_files = glob.glob(os.path.join(
            self._path_outputs, 'schout_0000_*.nc'))
        path_file = os.path.join(self._path_outputs, schout_files[0])
        with nc.Dataset(path_file) as root:
            i23d = getattr(root.variables[name], "i23d")
            ivs = getattr(root.variables[name], "ivs")
        if i23d == -1:
            raise RuntimeError("i23d is not retrieved")
        if nrec % skip != 0:
            raise ValueError(
                "The value of 'skip' should be a factor of the number of the records in a spool.")
        n_rows = n_spools * nrec // skip
        n_nodes_global = self.mesh.global_dims['np']
        n_vrt = self.mesh.global_dims['nvrt']
        if node_i is None:
            if i23d == 1:
                shape = (n_rows, n_nodes_global)
            elif i23d == 2:
                if ivs == 1:
                    shape = (n_rows, n_nodes_global, n_vrt)
                elif ivs == 2:
                    shape = (n_rows, n_nodes_global, n_vrt, 2)
            else:
                raise NotImplementedError("Data type not supported")
        else:
            raise NotImplementedError("")
        variable = np.empty(shape)
        for spool_i in range(spool_begin, spool_end + 1):
            slice_time = slice((spool_i - spool_begin) * nrec // skip,
                               (spool_i - spool_begin + 1) * nrec // skip)
            for proc_i in range(nproc):
                path_schout = os.path.join(self._path_outputs,
                                           'schout_{:04d}_{:d}.nc'.format(proc_i, spool_i))
                map_node_local_to_global = self.mesh.nodes_local_to_global[proc_i][:, 1]
                with nc.Dataset(path_schout) as root:
                    nc_var = root.variables[name]
                    # 2D variable (time, nodes)
                    if i23d == 1:
                        if ivs == 1:
                            variable[slice_time] = nc_var[::skip, :]
                        else:
                            raise NotImplementedError("i23d=1, ivs=2 not impelmeneted")
                    # 3D variable (time, nodes, levels)
                    elif i23d == 2:
                        # Scalar
                        if ivs == 1:
                           variable[slice_time,
                                     map_node_local_to_global, :] = \
                                             nc_var[::skip, :, :]
                        # Vector
                        elif ivs == 2:
                            variable[slice_time,
                                     map_node_local_to_global, :] = \
                                nc_var[::skip, :, :]
                    # 4D variable
                    elif i23d == 3:
                        if ivs == 1:
                            if level is None:
                                variable[slice_time,
                                         map_node_local_to_global, :, :] = \
                                    nc_var[::skip, :, :]
                            else:
                                variable[slice_time,
                                         map_node_local_to_global, :] = \
                                    nc_var[::skip, :, level]
                        elif ivs == 2:
                            if level is None:
                                variable[slice_time,
                                         map_node_local_to_global, :, :, :] = \
                                    nc_var[::skip, :, :, :]
                            else:
                                variable[slice_time,
                                         map_node_local_to_global, :, :] = \
                                    nc_var[::skip, :, level, :]
        return variable

    def find_local_node_index(self, global_node_i):
        """ Find the local node index and the rank from a global node index.

            Parameters
            ----------
            global_node_i: int
                global node index to search (zero-based)

            Returns
            -------
            int
                rank where the node belongs to (zero-based)
            int
                local node index in the rank (zero-based)
        """
        for rank in range(self.mesh.global_dims['nproc']):
            result = np.where(
                self.mesh.nodes_local_to_global[rank][:, 1] == global_node_i)
            if result[0].size != 0:
                return rank, result[0][0]
        raise ValueError("Cannot find the node")


def read_schout(path_outputs_dir, *args, **kwargs):
    """
    Create a reader object from the study path to read schout files

    Parameters
    ----------
    path_outputs_dir: str
        The outputs path of a SCHISM study

    Returns
    -------
    """
    schout = Schout(path_outputs=path_outputs_dir)
    return schout
