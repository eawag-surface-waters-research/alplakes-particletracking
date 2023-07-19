import os
import shutil
import numpy as np
from functions import download_simulation_data, flip_list_dict, d3d_mitgcm_velocity_converter, replace_string


def preprocess(run_id, lake, start_time, end_time, particles,
               api="https://alplakes-api.eawag.ch/simulations/file/delft3d-flow/{}/{}",
               dt=30,
               out_dt=3600):
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    working_dir = os.path.join(root, "runs", "{}_{}_{}_{}particles_{}".format(lake, start_time.strftime('%Y%m%d%H%M'),
                                                                              end_time.strftime('%Y%m%d%H%M'),
                                                                              len(particles), run_id))
    os.makedirs(working_dir, exist_ok=True)

    p = flip_list_dict(particles)
    particles_dir = os.path.join(working_dir, "particles.npz")
    np.savez(particles_dir, x_seed=p["x"], y_seed=p["y"], z_seed=p["z"])

    velocity_field_dir = os.path.join(working_dir, "velocity_field")
    shutil.copytree(os.path.join(root, "grids", lake), velocity_field_dir, dirs_exist_ok=True)

    files = download_simulation_data(start_time, end_time, lake, api, os.path.join(root, "data", "simulations"))
    for file in files:
        d3d_mitgcm_velocity_converter(file, velocity_field_dir, dt)

    configuration_file = os.path.join(working_dir, "configuration.py")
    shutil.copyfile(os.path.join(root, "ctracker", "configuration_template.py"), configuration_file)

    replace_string(configuration_file, "$gcm_directory", velocity_field_dir)
    replace_string(configuration_file, "$gcm_dt", str(dt))
    replace_string(configuration_file, "$out_dt", str(out_dt))
    replace_string(configuration_file, "$outfile", os.path.join(working_dir, "output", "results.nc"))
    replace_string(configuration_file, "$particles", particles_dir)
    replace_string(configuration_file, "$start", start_time.strftime("%Y-%m-%d %H:%M"))
    replace_string(configuration_file, "$end", end_time.strftime("%Y-%m-%d %H:%M"))
