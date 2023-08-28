import os
import netCDF4
import shutil
import random
import requests
import xarray as xr
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.path import Path
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


def convert_to_grid_coordinates(particles, x0, y0, x1, y1):
    grid_angle = np.arctan2(y1 - y0, x1 - x0)
    for p in particles:
        x = np.cos(grid_angle) * (p["x"] - x0) + np.sin(grid_angle) * (p["y"] - y0)
        y = -np.sin(grid_angle) * (p["x"] - x0) + np.cos(grid_angle) * (p["y"] - y0)
        p["x"] = x
        p["y"] = y
    return particles


def random_point_in_circle(x, y, radius):
    r = radius * random.random()
    theta = 2 * np.pi * random.random()
    return x + r * np.cos(theta), y + r * np.sin(theta)


def random_points_in_circle(x, y, radius, n, min_z, max_z):
    particles = []
    for i in range(n):
        zz = random.uniform(min_z, max_z)
        xx, yy = random_point_in_circle(x, y, radius)
        particles.append({"x": xx, "y": yy, "z": zz})
    return particles


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
    dx0_const = np.nanmean(np.diff(xG[1, :]))  # distance between two grid points along axis=1
    dx1_const = np.nanmean(np.diff(xG[:, 1]))  # distance between two grid points along axis=0
    dgrid_const = np.sqrt((dx0_const) ** 2 + (dx1_const) ** 2)  # dx

    # expand the grid to a rectangle
    for ii in range(xG.shape[1] - 1):  # usually all the items in xG[:,-1] are nan
        ind_strt = np.argwhere(~np.isnan(xG[:, ii]))[0][0]
        ind_end = np.argwhere(~np.isnan(xG[:, ii]))[-1][0]

        xG[0, ii] = xG[ind_strt, ii] - dx1_const * ind_strt
        for jj in range(1, xG.shape[0]):
            xG[jj, ii] = xG[jj - 1, ii] + dx1_const

    xG[:, -1] = xG[:, -2] + dx0_const

    # repeat the same procedure for dy and y grid

    dx0_const = np.nanmean(np.diff(yG[1, :]))
    dx1_const = np.nanmean(np.diff(yG[:, 1]))
    dgrid_const = np.sqrt((dx0_const) ** 2 + (dx1_const) ** 2)

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

    ind_surface = np.where(ds.ZK.data > 0)[0][0] - 1

    ds.close()

    return xG, yG, xC, yC, dgrid_const, X0, Y0, X1, Y1, ind_surface


def d3d_mitgcm_velocity_converter(file, output_folder, dt):
    xG, yG, xC, yC, dgrid_const, x0, y0, x1, y1, ind_surface = extract_grid_d3d(file)
    with netCDF4.Dataset(file, 'r') as nc:
        time = nc.variables['time'][:]
        for i in range(len(time)):
            uvel_dum = nc.variables['U1'][i, :ind_surface + 1, :xG.shape[1], :xG.shape[0]]
            u_data = uvel_dum[::-1, :, :]
            u_data[(u_data < -10) | (u_data > 10)] = 0
            vvel_dum = nc.variables['V1'][i, :ind_surface + 1, :xG.shape[1], :xG.shape[0]]
            v_data = vvel_dum[::-1, :, :]
            v_data[(v_data < -10) | (v_data > 10)] = 0
            wvel_dum = nc.variables['WPHY'][i, :ind_surface + 1, :xG.shape[1], :xG.shape[0]]
            w_data = wvel_dum[::-1, :, :]
            w_data[(w_data < -10) | (w_data > 10)] = 0

            # Re-arrange the dimensions to match the UVEL, VVEL, and WVEL variables in a MITgcm output netCDF file
            uvel_data = u_data.transpose((0, 2, 1))
            vvel_data = v_data.transpose((0, 2, 1))
            wvel_data = w_data.transpose((0, 2, 1))

            # Combine the velocity components into a single 4D array with dimensions (3, nx, ny, nz)
            uvw_data = np.zeros((3, uvel_data.shape[0], uvel_data.shape[1], uvel_data.shape[2]), dtype=np.float32)

            uvw_data[0, :, :, 1:uvel_data.shape[2]] = uvel_data[:, :, 0:uvel_data.shape[2] - 1]
            uvw_data[1, :, 1:uvel_data.shape[1], :] = vvel_data[:, 0:uvel_data.shape[1] - 1, :]
            uvw_data[2, :, :, :] = wvel_data

            wrmds(output_folder + '/3Dsnaps', uvw_data, ndims=[3],
                  dimlist=[uvel_data.shape[2], uvel_data.shape[1], uvel_data.shape[0]], dataprec=['float32'],
                  nrecords=[3], times=time[i], fields=['UVEL', 'VVEL', 'WVEL'], deltat=dt, machineformat='l')
    return x0, y0, x1, y1


def replace_string(file_path, target_string, replacement_string):
    with open(file_path, 'r') as file:
        file_content = file.read()
    modified_content = file_content.replace(target_string, replacement_string)
    with open(file_path, 'w') as file:
        file.write(modified_content)


def plot_particle_tracking(working_dir, x0, y0, x1, y1,lake, save=False, bathy=False, plot=True,grid_type="cartesian"):
    data = xr.open_dataset(os.path.join(working_dir, "output", "results_run.nc"), decode_times=True)
    inout = xr.open_dataset(os.path.join(working_dir, "output", "results_inout.nc"), decode_times=True)

    time_plt = data.time
    depth = np.array(data.Depth[:])

    # Release depth which will be plotted
    z0_ini = [0, 30]
    zdata = []
    sel_pid = inout.pid.where((inout.z_ini > z0_ini[0]) & (inout.z_ini <= z0_ini[1]), drop=True)
    zd = data.sel(pid=sel_pid, drop=True)
    zd.xtrack.load()
    zd.ytrack.load()
    zd.ztrack.load()
    zdata.append(zd)

    # Convert grid to model co-ordinates
    xG = data.XG_p1
    yG = data.YG_p1
    
    if grid_type=="cartesian":
        gridAngle = np.arctan2(y1 - y0, x1 - x0)
        xg_plt = xG.to_masked_array().filled()
        yg_plt = yG.to_masked_array().filled()
        xp_conv = (np.cos(gridAngle) * xg_plt - np.sin(gridAngle) * yg_plt) + x0
        yp_conv = (np.sin(gridAngle) * xg_plt + np.cos(gridAngle) * yg_plt) + y0
        xp_conv *= 1.e-3
        yp_conv *= 1.e-3
        outline = np.array(data.hFacC[0, :])

        for qq in range(len(time_plt)):
            for zd in zdata:
                now = zd.sel(time=time_plt[qq], drop=True)
                xp = (now.xtrack).to_masked_array().filled()
                xp[xp > 1e9] = np.nan
                yp = now.ytrack.to_masked_array().filled()
                yp[yp > 1e9] = np.nan
                zp = now.ztrack.to_masked_array().filled()
                zp[zp > 450] = np.nan

            part_no_snap, _, _ = np.histogram2d(xp, yp, bins=[xg_plt[0, :], yg_plt[:, 0]])
            part_no_snap = part_no_snap.T
            part_no_snap[part_no_snap < 2] = np.nan
            min_val = 2
            part_no_snap[part_no_snap < min_val] = np.nan

            part_depth_snap = part_no_snap.copy()
            part_depth_snap[:, :] = 0

            for ii in range(len(xp)):
                if (~np.isnan(xp[ii])) & (~np.isnan(yp[ii])):
                    ind_x = np.searchsorted(0.5 * (xg_plt[0, :-1] + xg_plt[0, 1:]), xp[ii])
                    ind_y = np.searchsorted(0.5 * (yg_plt[:-1, 0] + yg_plt[1:, 0]), yp[ii])
                    if (~np.isnan(zp[ii])) & (zp[ii] < 5):
                        part_depth_snap[ind_y, ind_x] = part_depth_snap[ind_y, ind_x] + zp[ii]

            part_depth_snap = part_depth_snap / part_no_snap

            fig_size = (8, 8)
            f = plt.figure(figsize=fig_size)
            ax = f.add_subplot(121)

            if bathy:
                plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], depth, cmap='Greys', vmax=np.max(depth) * 1.5)
            else:
                plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], outline, cmap='Greys', vmax=10)
            plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], part_no_snap, vmin=np.nanquantile(part_no_snap, 0.25),
                           vmax=np.nanquantile(part_no_snap, 0.75), cmap='viridis')
            ax.set_aspect('equal')
            ax.set_xlabel("Lat (km CH1903)")
            ax.set_ylabel("Lon (km CH1903)")
            ax.annotate(str(time_plt[qq].values.astype("datetime64[m]")).replace("T", " "), xy=(0.02, 0.97),
                        xycoords='axes fraction')

            cbar = plt.colorbar(fraction=0.02, orientation="horizontal", pad=-0.1, extend='max')
            cbar.ax.set_xticklabels([])
            cbar.ax.set_xticks([])
            cbar.set_label(label='$\mathregular{Particle \ concentration\ [-]}$')

            ax2 = f.add_subplot(122)
            if bathy:
                plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], depth, cmap='Greys', vmax=np.max(depth) * 1.5)
            else:
                plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], outline, cmap='Greys', vmax=10)
            plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], part_depth_snap, cmap='jet', vmin=0, vmax=40)
            ax2.set_aspect('equal')
            ax2.set_xlabel("Lat (km CH1903)")
            ax2.axes.yaxis.set_ticklabels([])

            cbar2 = plt.colorbar(fraction=0.02, orientation="horizontal", pad=-0.1, extend='max', ticks=[0, 10, 20])
            cbar2.ax.set_xticklabels([0, 10, 20])
            cbar2.set_label(label='$\mathregular{Depth\ [m]}$')
            cbar2.ax.xaxis.set_ticks_position('top')

            if save:
                out_dir = os.path.join(working_dir, "plots")
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(out_dir + "/plot" + '_' + str(time_plt[qq].values.astype("datetime64[m]")).replace("T", " ")[
                                                      0:10] + 'H' + str(
                    time_plt[qq].values.astype("datetime64[m]")).replace("T", " ")[11:13] + '.png', dpi=300,
                            bbox_inches='tight')
            if plot:
                plt.show()

            plt.close()
            
    elif grid_type=="curvilinear":
        xx_com = np.array(xG.data)
        yy_com = np.array(yG.data)

        xG_conv = xx_com
        yG_conv = yy_com
        outline = np.array(data.hFacC[0, :])
        
        # create the grid for spatiotemporal plots
        ds_offset = 100
        ds_bin = 50
        xp_conv = xG.to_masked_array().filled()
        yp_conv = yG.to_masked_array().filled()
        x_arr = np.linspace((xp_conv[xp_conv>0].min()-ds_offset),(xp_conv[xp_conv>0].max()+ds_offset),np.ceil((xp_conv[xp_conv>0].max()-xp_conv[xp_conv>0].min())/ds_bin).astype(int))
        y_arr = np.linspace((yp_conv[yp_conv>0].min()-ds_offset),(yp_conv[yp_conv>0].max()+ds_offset),np.ceil((yp_conv[yp_conv>0].max()-yp_conv[yp_conv>0].min())/ds_bin).astype(int))
        xg_plt,yg_plt = np.meshgrid(x_arr,y_arr)
        
        root = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))
        xs, ys = np.genfromtxt(os.path.join(root, "lake_boundary", lake,"lake_boundary.ldb"),unpack=True)
        xs = xs[2:] # to remove the header from ldb file
        ys = ys[2:]
        poly_path=Path([(xs[ii],ys[ii]) for ii in range(len(xs))])
        coors=np.hstack((xg_plt[:-1,:-1].reshape(-1, 1), yg_plt[:-1,:-1].reshape(-1,1)))
        mask_lake = poly_path.contains_points(coors).reshape(xg_plt[:-1,:-1].shape)
        
        for qq in range(len(time_plt)):
            ###############
            for zd in zdata:
                # now = zd.sel(time=np.datetime64(cutoff_date), drop=True)
                now = zd.sel(time=time_plt[qq], drop=True)
                xp = (now.xtrack).to_masked_array().filled()
                xp[xp > 1e9] = np.nan
                yp = now.ytrack.to_masked_array().filled()
                yp[yp > 1e9] = np.nan
                zp = now.ztrack.to_masked_array().filled()
                zp[zp > 450] = np.nan

            ###############   
            part_no_snap, _, _ = np.histogram2d(xp, yp, bins=[xg_plt[0,:],yg_plt[:,0]])
            part_no_snap = part_no_snap.T
            min_val = 2
            part_no_snap[part_no_snap<min_val] = np.nan
            part_no_snap[~mask_lake] = np.nan
            
            ############### 
            part_depth_snap = part_no_snap.copy()
            part_depth_snap[:,:] = 0

            for ii in range(len(xp)):
                if ((~np.isnan(xp[ii])) & (~np.isnan(yp[ii]))):
                    ind_x = np.searchsorted(0.5*(xg_plt[0,:-1]+xg_plt[0,1:]),xp[ii])
                    ind_y = np.searchsorted(0.5*(yg_plt[:-1,0]+yg_plt[1:,0]),yp[ii])
                    if ((~np.isnan(zp[ii])) & (ind_x<part_depth_snap.shape[1]) & (ind_y<part_depth_snap.shape[0])):
                        part_depth_snap[ind_y,ind_x] = part_depth_snap[ind_y,ind_x]+zp[ii]

            part_depth_snap = part_depth_snap/part_no_snap
            # part_depth_snap[part_depth_snap>60] = np.nan
            # part_no_snap[part_depth_snap>10] = np.nan
            part_depth_snap[~mask_lake] = np.nan
            
            fig_size = (8, 8)
            f = plt.figure(figsize=fig_size)
            ax = f.add_subplot(121)

            if bathy:
                plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], depth, cmap='Greys', vmax=np.max(depth) * 1.5)
            else:
                plt.plot(xp_conv, yp_conv,'ko',markersize=1,alpha=0.1)
            plt.pcolormesh(xg_plt[:-1,:-1],yg_plt[:-1,:-1],part_no_snap, vmin=np.nanquantile(part_no_snap, 0.25),
                           vmax=np.nanquantile(part_no_snap, 0.75), cmap='viridis')
            plt.xlim([xs.min()-500,xs.max()+500])
            plt.ylim([ys.min()-500,ys.max()+500])
            ax.set_aspect('equal')
            ax.set_xlabel("Lat (km CH1903)")
            ax.set_ylabel("Lon (km CH1903)")
            ax.annotate(str(time_plt[qq].values.astype("datetime64[m]")).replace("T", " "), xy=(0.02, 0.97),
                        xycoords='axes fraction')

            cbar = plt.colorbar(fraction=0.02, orientation="horizontal", pad=-0.1, extend='max')
            cbar.ax.set_xticklabels([])
            cbar.ax.set_xticks([])
            cbar.set_label(label='$\mathregular{Particle \ concentration\ [-]}$')

            ax2 = f.add_subplot(122)
            if bathy:
                plt.pcolormesh(xp_conv[:-1, :-1], yp_conv[:-1, :-1], depth, cmap='Greys', vmax=np.max(depth) * 1.5)
            else:
                plt.plot(xp_conv, yp_conv,'ko',markersize=1,alpha=0.1)
            plt.pcolormesh(xg_plt[:-1,:-1],yg_plt[:-1,:-1],part_depth_snap, cmap='jet', vmin=0, vmax=40)
            
            plt.xlim([xs.min()-500,xs.max()+500])
            plt.ylim([ys.min()-500,ys.max()+500])
            ax2.set_aspect('equal')
            ax2.set_xlabel("Lat (km CH1903)")
            ax2.axes.yaxis.set_ticklabels([])

            cbar2 = plt.colorbar(fraction=0.02, orientation="horizontal", pad=-0.1, extend='max', ticks=[0, 10, 20])
            cbar2.ax.set_xticklabels([0, 10, 20])
            cbar2.set_label(label='$\mathregular{Depth\ [m]}$')
            cbar2.ax.xaxis.set_ticks_position('top')

            if save:
                out_dir = os.path.join(working_dir, "plots")
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(out_dir + "/plot" + '_' + str(time_plt[qq].values.astype("datetime64[m]")).replace("T", " ")[
                                                      0:10] + 'H' + str(
                    time_plt[qq].values.astype("datetime64[m]")).replace("T", " ")[11:13] + '.png', dpi=300,
                            bbox_inches='tight')
            if plot:
                plt.show()

            plt.close()
    
    data.close()
    inout.close()