import os
import netCDF4
import shutil
import requests
import xarray as xr
import numpy as np
from tqdm import tqdm
from MITgcmutils import mds, wrmds
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SU


def download_file(url, filename):
    response = requests.get(url, stream=True)

    # Check if the response is successful (status code 200)
    if response.status_code != 200:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")

    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def last_sunday(date):
    return date + relativedelta(weekday=SU(-1))


def download_simulation_data(start, end, lake, api, folder, max_iter=52):
    if end < start:
        raise ValueError("Start date must be before end date.")
    path = os.path.join(folder, lake)
    os.makedirs(path, exist_ok=True)
    date = last_sunday(start)
    files = [date]
    i = 0
    while date + timedelta(days=7) < end and i < max_iter:
        i = i + 1
        date = date + timedelta(days=7)
        files.append(date)

    file_paths = []
    for file in files:
        out_file = os.path.join(path, "{}.nc".format(file.strftime('%Y%m%d')))
        file_paths.append(out_file)
        if not os.path.exists(out_file):
            download_file(api.format(lake, file.strftime('%Y%m%d')), out_file)

    return file_paths


def flip_list_dict(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists


def extract_grid_d3d(file):
    ds = xr.open_dataset(file, engine="netcdf4")
    xG = ds.XCOR[:].data.T
    yG = ds.YCOR[:].data.T
    xG[xG <= 0] = np.nan
    yG[yG <= 0] = np.nan
    xC = ds.XZ[:].data.T
    yC = ds.YZ[:].data.T
    xC[xC <= 0] = np.nan
    yC[yC <= 0] = np.nan
    # Compute dx (here we assume dx=dy)
    dx0_const = np.nanmean(np.diff(xG[1, :]))  # distance between two grid points along axis=1
    dx1_const = np.nanmean(np.diff(xG[:, 1]))  # distance between two grid points along axis=0
    dgrid_const = np.sqrt(dx0_const ** 2 + dx1_const ** 2)  # dx

    # expand the grid to a rectangle
    for ii in range(xG.shape[1] - 1):  # usually all the items in xG[:,-1] are nan
        ind_strt = np.argwhere(~np.isnan(xG[:, ii]))[0][0]
        ind_end = np.argwhere(~np.isnan(xG[:, ii]))[-1][0]

        xG[0, ii] = xG[ind_strt, ii] - dx1_const * ind_strt
        for jj in range(1, xG.shape[0]):
            xG[jj, ii] = xG[jj - 1, ii] + dx1_const

    xG[:, -1] = xG[:, -2] + dx0_const

    dx0_const = np.nanmean(np.diff(yG[1, :]))
    dx1_const = np.nanmean(np.diff(yG[:, 1]))
    dgrid_const = np.sqrt(dx0_const ** 2 + dx1_const ** 2)

    for ii in range(yG.shape[1] - 1):  # usually all the items in yG[:,-1] ar nan
        ind_strt = np.argwhere(~np.isnan(yG[:, ii]))[0][0]
        ind_end = np.argwhere(~np.isnan(yG[:, ii]))[-1][0]

        yG[0, ii] = yG[ind_strt, ii] - dx1_const * ind_strt
        for jj in range(1, yG.shape[0]):
            yG[jj, ii] = yG[jj - 1, ii] + dx1_const

    yG[:, -1] = yG[:, -2] + dx0_const

    # rotation-related parameters (these are two large black points in the plot below)
    (X0, Y0) = (xG[0, 0], yG[0, 0])
    (X1, Y1) = (xG[0, -1], yG[0, -1])

    ind_surface = np.where(ds.ZK_LYR.data > 0)[0][0]

    ds.close()

    return xG, yG, xC, yC, dgrid_const, X0, Y0, X1, Y1, ind_surface


def d3d_mitgcm_velocity_converter(file, output_folder, dt):
    xG, yG, xC, yC, dgrid_const, X0, Y0, X1, Y1, ind_surface = extract_grid_d3d(file)
    with netCDF4.Dataset(file, 'r') as nc:
        time = nc.variables['time'][:]
        for i in range(len(time)):
            uvel_dum = nc.variables['U1'][i, :ind_surface, :xG.shape[1] - 1, :xG.shape[0] - 1]
            u_data = uvel_dum[::-1, :, :]
            vvel_dum = nc.variables['V1'][i, :ind_surface, :xG.shape[1] - 1, :xG.shape[0] - 1]
            v_data = vvel_dum[::-1, :, :]
            wvel_dum = nc.variables['WPHY'][i, :ind_surface, :xG.shape[1] - 1, :xG.shape[0] - 1]
            w_data = wvel_dum[::-1, :, :]

            # Re-arrange the dimensions to match the UVEL, VVEL, and WVEL variables in a MITgcm output netCDF file
            uvel_data = u_data.transpose((0, 2, 1))
            vvel_data = v_data.transpose((0, 2, 1))
            wvel_data = w_data.transpose((0, 2, 1))

            # Combine the velocity components into a single 4D array with dimensions (3, nx, ny, nz)
            uvw_data = np.zeros((3, uvel_data.shape[0], uvel_data.shape[1], uvel_data.shape[2]), dtype=np.float32)
            uvw_data[0, :, :, :] = uvel_data
            uvw_data[1, :, :, :] = vvel_data
            uvw_data[2, :, :, :] = wvel_data

            wrmds(output_folder + '/3Dsnaps', uvw_data, ndims=[3],
                  dimlist=[uvel_data.shape[2], uvel_data.shape[1], uvel_data.shape[0]], dataprec=['float64'],
                  nrecords=[3], times=time[i], fields=['UVEL', 'VVEL', 'WVEL'], deltat=dt, machineformat='b')


def replace_string(file_path, target_string, replacement_string):
    with open(file_path, 'r') as file:
        file_content = file.read()
    modified_content = file_content.replace(target_string, replacement_string)
    with open(file_path, 'w') as file:
        file.write(modified_content)

