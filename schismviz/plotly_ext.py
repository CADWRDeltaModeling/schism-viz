# uncomment and the run install script below if plotly is not available
#!conda install -y -c conda-forge plotly
import holoviews as hv
hv.extension('plotly')

import param

class TriSurface(hv.TriSurface):
    simplices = param.Array()

class TriSurfacePlot(hv.plotting.plotly.TriSurfacePlot):
    style_opts = ['cmap', 'plot_edges']

    def get_data(self, element, ranges, style, **kwargs):
        if element.simplices is None:
            return super(TriSurfacePlot, self).get_data(element, ranges, style, **kwargs)
        x, y, z = (element.dimension_values(i) for i in range(3))
        simplices = element.simplices
        return [dict(x=x, y=y, z=z, simplices=simplices)]
    
hv.Store.register({TriSurface: TriSurfacePlot}, 'plotly')
