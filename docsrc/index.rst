.. Schism Visualization using Python
   This repo houses notebooks and scripts to visualize SCHISM inputs and outputs

.. _getstart:

A work in progress...
=======================

This repo has a number of example notebooks which give simple examples. As those functionality evolves, the scripts in schism_viz
house common functions

The notebooks found in the `github repository`_ contain the following exercises:

.. _github repository:  https://github.com/CADWRDeltaModeling/schism-viz


.. code-block:: console

   conda env create --name schism_viz -f environment.yml

This creates a new environment called "schism_viz" which contains the necessary packages to run through these modules.

==================
Notebook Examples
==================

The data is from running `Module of 1 of HelloSchism <https://cadwrdeltamodeling.github.io/HelloSCHISM/html/helloschism.html>`_

 * `View mesh in 3D <_static/html/00_mesh_view_3D.html>`_
 .. image:: _static/images/3dmesh.gif
 * `Animate water surface elevation in 3D <_static/html/01_water_surface_elevation_animation.html>`_
 .. image:: _static/images/water_surface_3d_animation.gif
 * `Salinity animation with colors <_static/html/02_salinity_animation.html>`_
 .. image:: _static/images/salinity_color_animation.gif
 * `Velocity vectors with arrows and colors <_static/html/03_velocity_vectors.html>`_
 .. image:: _static/images/velocity_vector_arrow_color_animation.gif
 * `Overlaying the two animations above <_static/html/04_salinity_and_velocity_animation.html>`_
 .. image:: _static/images/velocity_salinity_animation.gif
 * `Represent mesh as a networkx graph <_static/html/07_networkx.html>`_
 * `Faster 2D mesh display <_static/html/08_view_mesh_fast.html>`_
 * `Stations Map with mesh <_static/html/09_stations_map.html>`_
   Overlaying the stations as points onto the 2D mesh display
.. toctree::
   :hidden:

   self


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
