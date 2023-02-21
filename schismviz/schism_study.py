#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from schimpy import schism_mesh

__all__ = ['read_schism_study']


class SchismStudy:
    """
    """

    def __init__(self, path_study, options=None):
        """
        Constructor of the study

        Parameters
        ----------
        options: , optional
            Not being used
        """
        self._path_study = path_study
        self.read(self._path_study)
        self._hgrid = None

    def read(self, path, options=None):
        self.path = path
        self._param_nml = self.read_param_nml()

    def read_param_nml(self):
        """ Read param.nml file into a namelist variable

            Parameters
            ----------
            fname: str, optional
                file name for param.nml.
        """
        import f90nml
        fname = os.path.join(self.path, 'param.nml')
        return f90nml.read(fname)

    @property
    def hgrid(self):
        if self._hgrid is None:
            fname = os.path.join(self.path, 'hgrid.gr3')
            self._hgrid = schism_mesh.read_mesh(fname)
        return self._hgrid


def read_schism_study(path, options={'SED': True}):
    """
    """
    study = SchismStudy(path, options)
    return study
