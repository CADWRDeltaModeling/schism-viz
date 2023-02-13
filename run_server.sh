pip install -e . --no-deps
cd notebooks
panel serve --address 0.0.0.0 --port 80 --allow-websocket-origin="*" 00_mesh_view_3D.ipynb 01_water_surface_elevation_animation.ipynb 02_salinity_animation.ipynb 03_velocity_vectors.ipynb 04_salinity_and_velocity_animation.ipynb 05_salinity_and_velocity_per_level_animation.ipynb 08_view_mesh_fast.ipynb 09_stations_map.ipynb 