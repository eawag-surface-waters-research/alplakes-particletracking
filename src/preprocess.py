import os
import shutil
import numpy as np
from functions import download_simulation_data, flip_list_dict, d3d_mitgcm_velocity_converter, replace_string, convert_to_grid_coordinates


def preprocess(run_id, lake, start_time, end_time, particles,
               api="https://alplakes-api.eawag.ch/simulations/file/delft3d-flow/{}/{}",
               simulation_timestep=30, simulation_output_timestep=10800, threads=8):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    working_dir = os.path.join(root, "runs", "{}_{}_{}_{}".format(lake, start_time.strftime('%Y%m%d%H%M'),
                                                                  end_time.strftime('%Y%m%d%H%M'),
                                                                  run_id))
    os.makedirs(working_dir, exist_ok=True)

    velocity_field_dir = os.path.join(working_dir, "velocity_field")
    shutil.copytree(os.path.join(root, "grids", lake), velocity_field_dir, dirs_exist_ok=True)

    files = download_simulation_data(start_time, end_time, lake, api, os.path.join(root, "data", "simulations"))
    for file in files:
        x0, y0, x1, y1, grid_type = d3d_mitgcm_velocity_converter(file, velocity_field_dir, simulation_timestep)

    particles_path = os.path.join(working_dir, "particles.npz")
    if grid_type == "cartesian":
        p = convert_to_grid_coordinates(particles, x0, y0, x1, y1)
        p = flip_list_dict(p)
    elif grid_type == "curvilinear":
        p = particles.copy()
        p = flip_list_dict(p)
    else:
        print("Error in grid type. It must be either cartesian or curvilinear")
    
    np.savez(particles_path, x_seed=p["x"], y_seed=p["y"], z_seed=p["z"])

    configuration_file = os.path.join(working_dir, "configuration.py")
    shutil.copyfile(os.path.join(root, "ctracker", "configuration_template.py"), configuration_file)

    os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)

    replace_string(configuration_file, "$gcm_geometry", grid_type)
    replace_string(configuration_file, "$gcm_directory", velocity_field_dir)
    replace_string(configuration_file, "$simulation_timestep", str(simulation_timestep))
    replace_string(configuration_file, "$simulation_output_timestep", str(simulation_output_timestep))
    replace_string(configuration_file, "$outfile", os.path.join(working_dir, "output", "results.nc"))
    replace_string(configuration_file, "$particles", particles_path)
    replace_string(configuration_file, "$start", start_time.strftime("%Y-%m-%d %H:%M"))
    replace_string(configuration_file, "$end", end_time.strftime("%Y-%m-%d %H:%M"))
    replace_string(configuration_file, "$threads", str(threads))
    print(working_dir)

    return {"working_dir": working_dir, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "grid_type": grid_type}
