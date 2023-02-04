# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import bokeh.models
import holoviews as hv
from holoviews.plotting.util import process_cmap
from holoviews.operation.datashader import datashade, rasterize
hv.extension('bokeh')    # noqa

from schimpy import split_quad
from . import schism_study
from . import schout_reader


class SchismMeshNew:
    def is_pure_tri(self):
        return True


class SchoutTriMeshBase:
    def __init__(self, path_study, path_outputs='outputs', use_datashade=True):
        self._path_study = path_study
        self._path_outputs = path_outputs
        self._use_datashade = use_datashade
        self._study = self._read_schism_study()
        self._get_params()
        self.schout = self.read_schout()
        mesh = self.schout.schism_mesh
        self.mesh = split_quad.split_all_quads(mesh)
        self._hv_elems = self.mesh._elems[:, :3]
        self._hv_nodes = hv.Points(self.mesh._nodes, vdims='v1')

    def _get_params(self):
        params = self._study._param_nml
        nspool = params['core']['nspool']
        ihfskip = params['core']['ihfskip']
        dt = params['core']['dt']
        start_year = params['opt']['start_year']
        start_month = params['opt']['start_month']
        start_day = params['opt']['start_day']
        start_hour = params['opt']['start_hour']
        self._time_basis = datetime.datetime(
            start_year, start_month, start_day, start_hour)
        self.schout_dt = datetime.timedelta(seconds=dt * nspool)
        self._steps_per_spool = ihfskip // nspool

    def _read_schism_study(self):
        study = schism_study.read_schism_study(self._path_study)
        return study

    def read_schout(self):
        schout = schout_reader.read_schout(os.path.join(
            self._path_study, self._path_outputs))
        return schout

    def create_timestamps_from_params(self, spool_begin, spool_end):
        timestamps = [self._time_basis + (i + 1) * self.schout_dt
                      for i in range(
            (spool_begin - 1) * self._steps_per_spool,
            spool_end * self._steps_per_spool)]
        return timestamps

    def create_timestamps_from_schout(self, spool_begin, spool_end, skip):
        times = []
        for i in range(spool_begin, spool_end + 1):
            fname_nc = 'schout_0000_{:d}.nc'.format(i)
            path_nc = os.path.join(os.path.join(
                self._path_study, self._path_outputs), fname_nc)
            with nc.Dataset(path_nc, 'r') as f:
                times.extend([self._time_basis + datetime.timedelta(seconds=int(s))
                              for s in f.variables['time'][::skip]])
        return times

    def read_data(self, variable_name, spool_begin, spool_end, skip=1):
        variable = self.schout.variable(
            variable_name, spool_begin, spool_end, skip=skip)
        return variable


class SchoutTriMesh(SchoutTriMeshBase):
    """ Create a holoview TriMesh from a Schout data

        Parameters
        ----------
        mesh:
            mesh data
    """

    def __init__(self, path_study, path_outputs='outputs'):
        super().__init__(path_study, path_outputs)

    def dynamic_mesh(self, index, level):
        dataarray = self._xa_variable
        if len(dataarray.coords) == 2:
            array = dataarray.sel(time=index).values
        else:
            array = dataarray.sel(time=index, level=level).values
        if self._hv_nodes.data.shape[0] != array.shape[0]:
            raise RuntimeError(
                "The shape of the data does not agree with the mesh data")
        self._hv_nodes.data[:, -1] = array
        mesh = hv.TriMesh((self._hv_elems, self._hv_nodes),
                          label=self._label).opts(colorbar=True)
        return mesh

    def _get_clims(self, clims):
        if clims is not None:
            if isinstance(clims, str):
                if clims == 'auto':
                    return (self._variable.min(), self._variable.max())
                else:
                    raise ValueError("Not supported clims")
            elif isinstance(clims, tuple):
                return clims
            else:
                raise ValueError("Not supported clims type.")

    def _create_colorbar(self, clims, cmap, label):
        spacing = np.linspace(0, 1, 64)[::-1][np.newaxis].transpose()
        colorbar = hv.Image(spacing, xdensity=1,
                            bounds=(-1, clims[0], 1, clims[1])).opts(cmap=cmap,
                                                                     toolbar=None, width=100,
                                                                     height=self._height,
                                                                     xaxis=None, ylabel=label)
        return colorbar

    def _create_trimesh(self, variable_name, spool_begin, spool_end,
                        skip=1, use_datashade=True, clims=None, label=None,
                        aspect=None):
        """ Create a TrimMesh
        """
        self._timestamps = self.create_timestamps_from_schout(spool_begin,
                                                              spool_end,
                                                              skip)
        self._variable_name = variable_name
        self._variable = self.read_data(variable_name, spool_begin, spool_end,
                                        skip=skip)

        self._dim_time = hv.Dimension('time', values=self._timestamps)
        var_rank = len(self._variable.shape)
        self._label = self._variable_name if label is None else label
        if var_rank == 2:
            self._xa_variable = xr.DataArray(data=self._variable,
                                             coords=[self._timestamps,
                                                     range(self._variable.shape[1])],
                                             dims=['time', 'node'])
            dm_mesh = hv.DynamicMap(lambda time: self.dynamic_mesh(time, -1),
                                    kdims=self._dim_time)
        elif var_rank == 3:
            self._xa_variable = xr.DataArray(data=self._variable,
                                             coords=[self._timestamps,
                                                     range(
                                                         self._variable.shape[1]),
                                                     range(self._variable.shape[2])],
                                             dims=['time', 'node', 'level'])
            self._xa_zcor = xr.DataArray(data=self.read_data('zcor',
                                                             spool_begin,
                                                             spool_end,
                                                             skip),
                                         coords=[self._timestamps,
                                                 range(
                                                     self._variable.shape[1]),
                                                 range(self._variable.shape[2])],
                                         dims=['time', 'node', 'level'])
            self._dim_level = hv.Dimension('level',
                                           values=list(range(self._variable.shape[2])))
            dm_mesh = hv.DynamicMap(self.dynamic_mesh,
                                    kdims=[self._dim_time, self._dim_level])
        else:
            raise NotImplementedError()
        self._range_x = (self.mesh._nodes[:, 0].min(),
                         self.mesh._nodes[:, 0].max())
        self._range_y = (self.mesh._nodes[:, 1].min(),
                         self.mesh._nodes[:, 1].max())
        if aspect is None:
            self._aspect = (self._range_y[1] - self._range_y[0]) / \
                (self._range_x[1] - self._range_x[0])
        else:
            self._aspect = aspect
        self._width = 800
        self._height = int(self._width * self._aspect)
        self._cmap = process_cmap('rainbow')

        self._node_size = 0

        hv.output(widget_location='top')

        if use_datashade:
            self._clims = self._get_clims(clims)
            if self._clims is None:
                if bokeh.__version__ < '2.2.3':
                    ds_mesh = datashade(dm_mesh,
                                        cmap=self._cmap)
                else:
                    ds_mesh = rasterize(dm_mesh).opts(
                        cmap=self._cmap,
                        colorbar=True,
                        cnorm="eq_hist")
            else:
                ds_mesh = rasterize(dm_mesh).opts(clim=self._clims,
                                                  cmap=self._cmap,
                                                  colorbar=True)
            ds_mesh.opts(hv.opts(width=self._width, height=self._height))
            return ds_mesh
        else:
            dm_mesh.opts(hv.opts.TriMesh(filled=True,
                                         edge_color='v1',
                                         tools=['hover'],
                                         node_size=self._node_size,
                                         cmap=self._cmap,
                                         width=self._width,
                                         height=self._height))
            return dm_mesh

    def create_plot(self, variable_name, spool_begin, spool_end,
                    skip=1, use_datashade=True,
                    clims='auto', label=None,
                    aspect=None):
        """ Create a holoviews plot from the object


            Parameters
            ----------
            variable_name: str
                the name of a variable to read and plot
            spool_begin: int
                the spool number to begin to read
            spool_end: int
                the spool number to end to read (inclusive)
            skip: int, optional
                skip size to read the time steps.
                If it is not given, every time steps in the files will be read
                in. The value should be an integer factor of the number of
                the records in each spool.
            use_datashade: boolean, optional
                A flag to use Datashader.
                If True, Datashader is used to speed up and aggregate the
                values for visualization. The default value is True.
                If False, Holoviews TriMesh is used, so it could be slow
                or does not work for a big mesh.
            clims: tuple or str, optional
                A range for the range of the color mapping. The default
                value is 'auto'.
                If a string value 'auto' is given, the range is set
                automatically from the min and max of the loaded data.
                If a tuple of a pair of the minimum and maxmimum given,
                the tuple is used to set the range of the color mapping.
                If the value is None, the color mapping follows the
                default behaviors from Holoviews. In the case of the Holoviews
                Datashader, the color range (or clims) is re-adjusted when
                the plot is updated or zoomed automatically. The normalization
                is based on historgram equalization, not linear, so a colorbar
                will not show. This way shows the local differences very well
                without washing out variations by extreme values.
            aspect: float, optional
                The aspect of the axis scaling, i.e. the ratio of y-unit to
                x-unit. If not provided, it will use the ratio from the
                data range.

            Return
            ------
            holoviews.TriMesh
                a holoviews TriMesh plot for Jupyter Notebook
        """
        self._hv_mesh = self._create_trimesh(variable_name,
                                             spool_begin, spool_end,
                                             skip=skip,
                                             use_datashade=use_datashade,
                                             clims=clims,
                                             label=label,
                                             aspect=aspect)
        return self._hv_mesh


class SchoutTriMeshPointSelect(SchoutTriMesh):
    def __init__(self, path_study, path_outputs='outputs'):
        super().__init__(path_study, path_outputs)

    def dynamic_point_timeseries(self, variable_name, spool_begin, spool_end,
                                 level, data):
        dim_var = hv.Dimension(variable_name)
        dim_time = hv.Dimension('Time')
        if len(data['x']) == 0:
            return hv.Curve((pd.to_datetime([self._time_basis]), [0]),
                            dim_time, dim_var).opts(width=800, height=300,
                                                    show_grid=True)
        pos = (data['x'][0], data['y'][0])
        schout = self.schout
        elem_i = schout.schism_mesh.find_elem(pos)
        if elem_i is None:
            return hv.Curve(([self._time_basis], [0])).opts(width=800, height=300)
        else:
            node_i_global = schout.schism_mesh.find_closest_nodes(pos, count=1)
            rank, node_local_i = schout.find_local_node_index(node_i_global)
            val = schout.node_value(
                variable_name, rank, node_local_i, spool_begin, spool_end)
            times = pd.to_datetime([self._time_basis + (i + 1) * self.schout_dt
                                    for i in range((spool_begin - 1) * self._steps_per_spool,
                                                   spool_end * self._steps_per_spool)])
            if len(val.shape) == 1:
                return hv.Curve((times, val), dim_time, dim_var).opts(width=800,
                                                                      height=300,
                                                                      show_grid=True)
            elif len(val.shape) == 2:
                return hv.Curve((times, val[:, level]), dim_time, dim_var).opts(width=800,
                                                                                height=300,
                                                                                show_grid=True)
            else:
                raise NotImplementedError()

    def dynamic_point_profile(self, variable_name, index, data):
        dim_var = hv.Dimension(variable_name)
        dim_z = hv.Dimension('z')
        if len(data['x']) == 0:
            return hv.Curve(([], []), dim_var, dim_z).opts(width=300,
                                                           height=300,
                                                           show_grid=True)

        schout = self.schout
        pos = (data['x'][0], data['y'][0])
        elem_i = schout.schism_mesh.find_elem(pos)
        if elem_i is None:
            return hv.Curve(([], []),
                            dim_var, dim_z).opts(width=300, height=300,
                                                 show_grid=True)
        else:
            node_i_global = schout.schism_mesh.find_closest_nodes(pos, count=1)
            rank, node_local_i = schout.find_local_node_index(node_i_global)
            zcor = self._xa_zcor.sel(time=index, node=node_i_global).values
            dim_z = hv.Dimension('z')
            val = self._xa_variable.sel(time=index, node=node_i_global).values
            return hv.Curve((val, zcor), dim_var, dim_z).opts(width=300,
                                                              height=300,
                                                              show_grid=True)

    def create_timeseries_plot_at_point(self, variable_name, spool_begin, spool_end):
        hover = bokeh.models.HoverTool(tooltips=[('date', '@{Time}{%F %T}'),
                                                 (variable_name, '@{0}'.format(variable_name))],
                                       formatters={'@{Time}': 'datetime'})
        var_rank = self.schout.variable_rank(variable_name)
        if var_rank == 2:
            dm = hv.DynamicMap(lambda data:
                               self.dynamic_point_timeseries(
                                   variable_name, spool_begin, spool_end, -1, data),
                               streams=[self._point_stream]).opts(tools=[hover],
                                                                  framewise=True)
        elif var_rank == 3:
            # self._dim_level2 = hv.Dimension('level2', values=list(
            #     range(self.schout.global_dims['nvrt'])))
            dm = hv.DynamicMap(lambda level, data:
                               self.dynamic_point_timeseries(
                                   variable_name, spool_begin, spool_end,
                                   level, data),
                               kdims=self._dim_level,
                               streams=[self._point_stream]).opts(tools=[hover],
                                                                  framewise=True)
        else:
            raise NotImplementedError()
        return dm

    def create_profile_plot_at_point(self, variable_name):
        var_rank = self.schout.variable_rank(variable_name)
        if var_rank == 2:
            raise RuntimeError()
        elif var_rank == 3:
            dm = hv.DynamicMap(lambda index, data:
                               self.dynamic_point_profile(
                                   variable_name, index, data),
                               kdims=self._dim_time, streams=[self._point_stream]).opts(tools=['hover'], framewise=True)
        else:
            raise NotImplementedError()
        return dm

    def create_select_points(self):
        selected_points = hv.Points(([], [], []),
                                    vdims='z').redim.range(x=self._range_x,
                                                           y=self._range_y)
        point_stream = hv.streams.PointDraw(
            data=selected_points.columns(), num_objects=1, source=selected_points)
        return selected_points, point_stream

    def create_plot(self, variable_name, spool_begin, spool_end,
                    point_spool_begin=None, point_spool_end=None,
                    use_datashade=True, clims='auto', label=None,
                    aspect=None):
        if point_spool_begin is None:
            point_spool_begin = spool_begin
        if point_spool_end is None:
            point_spool_end = spool_end

        self._hv_mesh = self._create_trimesh(variable_name,
                                             spool_begin, spool_end,
                                             use_datashade=use_datashade,
                                             clims=clims, label=label)
        self._selected_points, self._point_stream = self.create_select_points()

        mesh_with_point = (self._hv_mesh * self._selected_points).opts(
            hv.opts.Layout(merge_tools=False),
            hv.opts.Points(active_tools=['point_draw']))
        self._hv_timeseries = self.create_timeseries_plot_at_point(
            variable_name, point_spool_begin, point_spool_end)
        var_rank = self.schout.variable_rank(variable_name)

        if var_rank == 2:
            return (mesh_with_point + self._hv_timeseries).opts(shared_axes=False).cols(1)
        elif var_rank == 3:
            self._hv_profile = self.create_profile_plot_at_point(variable_name)
            return (mesh_with_point +
                    self._hv_timeseries +
                    self._hv_profile).opts(shared_axes=False).cols(1)
        else:
            raise NotImplementedError()


class SchoutTriMeshPathSelect(SchoutTriMesh):
    def __init__(self, path_study, path_outputs='outputs'):
        super().__init__(path_study, path_outputs)

    def create_empty_cross_section(self):
        s = np.zeros((2, 2))
        return hv.QuadMesh((s, s, s),
                           kdims=[self._dim_s, self._dim_z],
                           vdims=self._dim_var).opts(width=800,
                                                     height=300,
                                                     cmap=self._cmap,
                                                     show_grid=True)

    def get_elem_i_from_path(self, vertices):
        elems = []
        for vert in vertices:
            elem_i = self.schout.schism_mesh.find_elem(vert)
            if elem_i is None:
                return None
            else:
                elems.append(elem_i)
        return elems

    def find_nodes_from_path(self, vertices):
        nodes = []
        for vert in vertices:
            node_i = self.schout.schism_mesh.find_closest_nodes(vert, count=1)
            nodes.append(node_i)
        return nodes

    def dynamic_path_profile(self, variable_name, index, data, clims):
        self._dim_s = hv.Dimension('s')
        self._dim_z = hv.Dimension('z')
        self._dim_var = hv.Dimension(variable_name)
        if data is None or len(data['xs'][0]) < 1:
            return self.create_empty_cross_section()

        vert_x = data['xs'][0]
        vert_y = data['ys'][0]
        verts = list(zip(vert_x, vert_y))
        elems = self.get_elem_i_from_path(verts)
        if elems is None:
            return self.create_empty_cross_section()

        nodes = self.find_nodes_from_path(verts)
        zs = self._xa_zcor.sel(time=index, node=nodes).values

        s = [0., ]
        for i in range(1, len(vert_x)):
            len_segment = np.linalg.norm([vert_x[i] - vert_x[i - 1],
                                          vert_y[i] - vert_y[i - 1]])
            s.append(len_segment + s[i - 1])
        ss = np.broadcast_to(np.array(s).reshape(-1, 1), (len(s), zs.shape[1]))

        vals = self._xa_variable.sel(time=index, node=nodes).values

        if False:
            pass
        else:
            return hv.QuadMesh((ss, zs, vals),
                               kdims=[self._dim_s, self._dim_z],
                               vdims=self._dim_var).opts(width=800,
                                                         height=300,
                                                         cmap=self._cmap,
                                                         show_grid=True)

    def create_crosssection_along_path(self, variable_name, clims):
        var_rank = self.schout.variable_rank(variable_name)
        if var_rank == 2:
            raise RuntimeError()
        elif var_rank == 3:
            dm = hv.DynamicMap(lambda index, data:
                               self.dynamic_path_profile(
                                   variable_name, index, data, clims),
                               kdims=self._dim_time,
                               streams=[self._path_stream]).opts(tools=['hover'],
                                                                 framewise=True)
        else:
            raise NotImplementedError()
        return dm

    def create_select_path(self):
        selected_path = hv.Path(([[]])).redim.range(x=self._range_x,
                                                    y=self._range_y)
        path_stream = hv.streams.PolyDraw(
            num_objects=1, source=selected_path, show_vertices=True)
        return selected_path, path_stream

    def create_plot(self, variable_name, spool_begin, spool_end,
                    point_spool_begin=None, point_spool_end=None,
                    use_datashade=True, clims='auto', label=None):
        if point_spool_begin is None:
            point_spool_begin = spool_begin
        if point_spool_end is None:
            point_spool_end = spool_end

        self._hv_mesh = self._create_trimesh(variable_name,
                                             spool_begin, spool_end,
                                             use_datashade=use_datashade,
                                             clims=clims,
                                             label=label)
        self._selected_path, self._path_stream = self.create_select_path()

        mesh_with_path = (self._hv_mesh * self._selected_path).opts(
            hv.opts.Layout(merge_tools=False),
            hv.opts.Path(color='red'))
        var_rank = self.schout.variable_rank(variable_name)

        if var_rank == 2:
            raise ValueError("2D data is not supported in this plot.")
        elif var_rank == 3:
            self._hv_profile = self.create_crosssection_along_path(
                variable_name, self._clims)
            return (mesh_with_path +
                    self._hv_profile).opts(shared_axes=False).cols(1)
        else:
            raise NotImplementedError()


# def create_holoviews_schout(path_study, path_outputs='outputs',
#                             use_datashade=True):
#     mesh = SchoutTriMesh(path_study, path_outputs)
#     trimesh = mesh.create_plot(use_datashade=use_datashade)
#     return trimesh
