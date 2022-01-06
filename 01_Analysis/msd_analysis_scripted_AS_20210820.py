# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:42:33 2021

@author: axel4
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
import fnmatch
import cycler


def read_tracks(filepath):
    # reads the .tck files
    with open(filepath) as fp:
        x = fp.readline().strip().replace(",", ".").split('\t')
        y = fp.readline().strip().replace(",", ".").split('\t')
        z = fp.readline().strip().replace(",", ".").split('\t')
        t = fp.readline().strip().replace(",", ".").split('\t')

    # XYZ - in microns, t - time in seconds. t0 is not equal to zero, so you might want to subtract it from every value in t
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    z = [float(i) for i in z]
    t = [float(i) for i in t]

    # norming the time
    t = t - np.min(t)
    t = t + t[1]
    # if for whatever reason time is given in milliseconds, here's the fix
    # t = t / 1000

    # getting the size of the dataset
    N = np.size(t)
    N = N - 1
    return x, y, z, t, N


def calc_and_plot_3d_track(x, y, z, t, N, name, foldername):
    XMin = np.min(x)
    XMax = np.max(x)

    YMin = np.min(y)
    YMax = np.max(y)

    ZMin = np.min(z)
    ZMax = np.max(z)

    # rough estimate of the volume explored by the particle
    Volume = (XMax - XMin) * (YMax - YMin) * (ZMax - ZMin)
    Volume = round(Volume, 2)

    # plotting the trajectory in 3D
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection" : "3d"},  dpi=300)
    ax.plot(x[1:N], y[1:N], z[1:N])
    ax.scatter3D(x[1:N], y[1:N], z[1:N], c=t[1:N], cmap='Greens', alpha=1, zorder=10)

    ax.set_xlabel('X, ${\mu}$m', fontsize=12)
    ax.set_ylabel('Y, ${\mu}$m', fontsize=12)
    ax.set_zlabel('Z, ${\mu}$m', fontsize=12)
    plt.subplots_adjust(top=1.1, left=0, bottom=0)

    plt.savefig(foldername + "/figures/" + "Trajectory " + str(name) + " 3Dplot.png", format="png")
    # SHOWPLOTS is set further
    if SHOWPLOTS:
        plt.show()
    plt.close()
    return Volume


def calc_and_plot_MSDs_XYZ(x, y, z, t, N):
    # Calculates MSD curves separately along each of the axes
    
    # for the MSD in X
    MSDinX = np.array(x[0:N])
    DispX = np.zeros((N, N)).astype('float64')
    for k in range(N):
        for i in range(N - k):
            DispX[k, i] = (x[i] - x[i + k]) ** 2

    for j in range(N):
        MSDinX[j] = np.sum(DispX[j, 0:N - j]) / (N - j)

    # plt.figure('Dispx')
    # plt.imshow(DispX)
    # plt.figure('MSD', dpi=300)
    # plt.plot(t[0:N - 1], MSDinX[1:N], 'xg-')  # green 'X' mark the MSD in X-direction
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim([0.5, N])
    # plt.ylim([0.002, 240])
    # if SHOWPLOTS:
    #     plt.show()
    # plt.close()
    
    # # for the MSD in Y
    MSDinY = np.array(y[0:N])
    DispY = np.zeros((N, N)).astype('float64')
    for k in range(N):
        for i in range(N - k):
            DispY[k, i] = (y[i] - y[i + k]) ** 2

    for j in range(N):
        MSDinY[j] = np.sum(DispY[j, 0:N - j]) / (N - j)
    # # plt.figure('Dispy')
    # # plt.imshow(DispY)
    # plt.figure('MSD', dpi=300)
    # plt.plot(t[0:N - 1], MSDinY[1:N], 'yd-')  # 'Y'ellow diamonds mark the MSD in Y direction
    # plt.xscale('log')
    # plt.yscale('log')
    # if SHOWPLOTS:
    #     plt.show()
    # plt.close()
    
    # for the MSD in Z
    MSDinZ = np.array(z[0:N])
    DispZ = np.zeros((N, N)).astype('float64')
    for k in range(N):
        for i in range(N - k):
            DispZ[k, i] = (z[i] - z[i + k]) ** 2

    for j in range(N):
        MSDinZ[j] = np.sum(DispZ[j, 0:N - j]) / (N - j)
    # # plt.figure('Dispz')
    # # plt.imshow(DispZ)
    # plt.figure('MSD', dpi=300)
    # plt.plot(t[0:N - 1], MSDinZ[1:N], 'ms-')  # magenta square'Z' mark the MSD in Z direction
    # plt.xscale('log')
    # plt.yscale('log')
    # #   plt.xlim([0.5, 6000]);
    # # plt.ylim([0.025, 300])
    # if SHOWPLOTS:
    #     plt.show()
    # plt.close()

    MSDinX = MSDinX[:-2]
    MSDinY = MSDinY[:-2]
    MSDinZ = MSDinZ[:-2]
    return MSDinX, MSDinY, MSDinZ


def calc_displacements_and_speeds(x, y, z, t, N):
    # Calculates displacements and momentary speeds
    # in X
    dx0 = np.array(x[0: N])
    dx1 = np.array(x[1: (N + 1)])
    dx = dx1 - dx0
    # in Y
    dy0 = np.array(y[0: N])
    dy1 = np.array(y[1: (N + 1)])
    dy = dy1 - dy0
    # in Z
    dz0 = np.array(z[0: N])
    dz1 = np.array(z[1: (N + 1)])
    dz = dz1 - dz0
    T = np.max(t)
    timeinterval = T / N
    dd = ((dx + dy + dz) ** 2) ** 0.5  # This is the displacement per step in 3D

    speeds = dd / timeinterval

    return dx, dy, dz, dd, speeds


def calc_MSD_3D(MSDinX, MSDinY, MSDinZ, t, N, name, foldername):
    # Calculates the full three-dimensional MSD and shows it in log-log scale
    DinXlist = []
    DinYlist = []
    DinZlist = []
    Din3Dlist = []

    MSD3D = (MSDinX + MSDinY + MSDinZ)
    plt.figure('Disp3D'); plt.imshow(Disp3D)
    plt.figure('MSD', dpi=300)
    plt.plot(t[0:N - 3], MSD3D[1:N], '--b')  # a dashed blue line shows the MSD for 3D movement
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time step, s')
    plt.ylabel('MSD, ${\mu}$m${^2}$/s')
    
    MSDxy = (MSDinX + MSDinY)
    plt.figure('MSD', dpi=300)
    plt.plot(t[0:N - 3], MSDxy[1:N], '-k')  # a plain black line shows the MSD for 3D movement
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([t[0], t[N - 1]], [MSDxy[1], MSDxy[1] * (N - 1)], 'k:')

    D3D = MSD3D[1] / (6 * t[0])
    Dxy = MSDxy[1] / (4 * t[0])
    Dx = MSDinX[1] / (2 * t[0])
    Dy = MSDinY[1] / (2 * t[0])
    Dz = MSDinZ[1] / (2 * t[0])

    plt.savefig(foldername + "/figures/" + "MSDs " + str(name) + " .png", format="png")
    if SHOWPLOTS:
       plt.show()
    plt.close()

    Din3Dlist = []
    local3Dlist = [str(name), D3D, ' um^2/s']
    Din3Dlist.append(local3Dlist)
    localxlist = [str(name), Dx, ' um^2/s']
    DinXlist.append(localxlist)
    localylist = [str(name), Dy, ' um^2/s']
    DinYlist.append(localylist)
    localzlist = [str(name), Dz, ' um^2/s']
    DinZlist.append(localzlist)
    return MSD3D, D3D, Dx, Dy, Dz, MSDxy, Dxy


def plot_speeds(speeds, t, N, name, foldername):
    # now just the speeds in windows of 10
    RS = np.linspace(1, N - 40, num=N - 40)
    # this gives a plot of the speeds the tracked particle has. Speeds are smoothed over a rolling window of 40 values.
    plt.figure('Rolling window speeds ' + str(name), dpi=300)
    for k in range(1):
        for i in range(N - 40):
            RS[i] = np.average(speeds[i: i + 40])
    plt.plot(t[0:N - 40], RS)
    plt.xlabel('time [s]')
    plt.ylabel('v [${\mu}$m/s]')
    RSMean = np.mean(RS)
    plt.plot([t[0], t[N - 40]], [RSMean, RSMean], ':k')  # this marks the mean of the speeds
    plt.plot([t[0], t[N - 40]], [1.15 * RSMean, 1.15 * RSMean],
             '-^g')  # draws a line with green upward pointing triangles for speeds that are 15% above the mean
    plt.plot([t[0], t[N - 40]], [0.85 * RSMean, 0.85 * RSMean],
             '-dm')  # draws a line with magenta diamonds for speeds that are 15% below the mean
    if SHOWPLOTS:
        plt.show()
    plt.close()
    
    plt.figure('Displacement speeds histogram ' + str(name), dpi=300)
    plt.title('Displacement speeds histogram ' + str(name))
    plt.hist(speeds, bins=20, range=[0, np.max(speeds)])
    plt.xlabel('v [${\mu}$m/s]')
    plt.ylabel('counts')
    plt.savefig(foldername + "/figures/" + "Displacement speeds histogram " + str(name) + " .png", format="png")
    if SHOWPLOTS:
        plt.show()
    plt.close()


def plot_2D_trajectories(x, y, z, N, name, foldername):
    # Plots the 2D projections of the entire trajectory
    # xy-Trajectory

    plt.figure('xy trajectory' + str(name), dpi=300)
    plt.title('NFD trajectory, ' + str(N) + ' steps;')
    plt.xlabel('x [${\mu}$m]')
    plt.ylabel('y [${\mu}$m]')
    plt.plot(x, y, 'r')  # the xy trajectory will be shown in red
    plt.savefig(foldername + "/figures/" + "X-Y trajectory" + str(name) + " with speeds.png", format="png")
    if SHOWPLOTS:
        plt.show()
    plt.close()
    # xz-Trajectory

    plt.figure('xz trajectory' + str(name), dpi=300)  # generates a figure that shows us the particles trajectory
    plt.title('NFD trajectory, ' + str(N) + ' steps')
    plt.xlabel('x [${\mu}$m]')
    plt.ylabel('z [${\mu}$m]')
    plt.plot(x, z, 'b')  # the xz trajectory will be shown in blue
    plt.savefig(foldername + "/figures/" + "X-Z trajectory" + str(name) + " with speeds.png", format="png")
    if SHOWPLOTS:
        plt.show()
    plt.close()
    # yz-Trajectory

    plt.figure('yz trajectory' + str(name), dpi=300)  # generates a figure that shows us the particles trajectory
    plt.title('NFD trajectory, ' + str(N) + ' steps')
    plt.plot(y, z, 'k')  # the yz trajectory will be shown in blue
    plt.xlabel('y [${\mu}$m]')
    plt.ylabel('z [${\mu}$m]')
    plt.savefig(foldername + "/figures/" + "Y-Z trajectory" + str(name) + " with speeds.png", format="png")
    if SHOWPLOTS:
        plt.show()
    plt.close()


def analyze_trajectory(filepath, name):
    # Analyzes the given trajectory, using the pre-defined functions
    x, y, z, t, N = read_tracks(filepath)
    Volume = calc_and_plot_3d_track(x, y, z, t, N, name, foldername)
    MSDinX, MSDinY, MSDinZ = calc_and_plot_MSDs_XYZ(x, y, z, t, N)
    dx, dy, dz, dd, speeds = calc_displacements_and_speeds(x, y, z, t, N)
    MSD3D, D3D, Dx, Dy, Dz, MSDxy, Dxy = calc_MSD_3D(MSDinX, MSDinY, MSDinZ, t, N, name, foldername)
    plot_speeds(speeds, t, N, name, foldername)
    plot_2D_trajectories(x, y, z, N, name, foldername)
    return (t, MSDinX, MSDinY, MSDinZ, MSD3D, D3D, Dx, Dy, Dz, MSDxy, Dxy, speeds, Volume)


def analyze_folder(dirpath):
    # Analyzes all files in a directory and creates a report
    MSD3D_max = []
    MSD3D_0 = []
    D3Ds = []
    DXs = []
    DYs = []
    DZs = []
    MSDxys = []
    Dxys = []
    speeds_median = []
    volumes = []
    files_path = [os.path.join(dirpath, x) for x in os.listdir(dirpath)]
    files_path.sort()
    batch_msds = [[] for i in range(len(fnmatch.filter(os.listdir(dirpath), '*.tck')))]
    bm_counter = 0
    batch_msds_xy = [[] for i in range(len(fnmatch.filter(os.listdir(dirpath), '*.tck')))]
    bm_xy_counter = 0
    for filepath in files_path:
        name = os.path.basename(filepath)
        if ".tck" in name:
            t, MSDinX, MSDinY, MSDinZ, MSD3D, D3D, Dx, Dy, Dz, MSDxy, Dxy, speeds, Volume = analyze_trajectory(filepath,
                                                                                                               name)
            MSD3D_max.append(MSD3D[-1])
            MSD3D_0.append(MSD3D[1])
            D3Ds.append(D3D)
            DXs.append(Dx)
            DYs.append(Dy)
            DZs.append(Dz)
            MSDxys.append(MSDxy)
            Dxys.append(Dxy)
            speeds_median.append(np.median(speeds))
            volumes.append(Volume)
            if BATCHMSD:
                batch_msds[bm_counter] = MSD3D
                bm_counter += 1
            if BATCHMSD_XY:
                batch_msds_xy[bm_xy_counter] = MSDxy
                bm_xy_counter += 1
    return t, MSD3D_max, MSD3D_0, D3Ds, DXs, DYs, DZs, MSDxys, Dxys, speeds_median, volumes, batch_msds, batch_msds_xy


# SHOWPLOTS: shows plots in the process, if true
# PRINTRESULTS: prints the results in the console, if true
# SAVERESULTS: writes the results in a txt file, if true
# BATCHMSD: creates the plot rendering all MSD curves, their median and interquartile range
# BATCHMSD_XY: same, but the MSDs are taken in XY only
# foldername: the directory with files. Only .tck files will be analyzed
SHOWPLOTS = False
PRINTRESULTS = False
SAVERESULTS = True
BATCHMSD = True
BATCHMSD_XY = False
foldername = "G:\My Drive\Work\!Work\Experimental data\Alina_T1 mapping\HeLa overnight\HeLa overnight_120nm_incubation\HeLa_overnight_120nm_incubation_24h_01_P01\\trajectories\good ones\merged\\21 minutes\\"

if os.path.exists(foldername + "/figures/") == False:
    os.makedirs(foldername + "/figures/")

# run the analysis and print the results
t, MSD3D_max, MSD3D_0, D3Ds, DXs, DYs, DZs, MSDSxys, Dxys, speeds_median, volumes, batch_msds, batch_msds_xy = analyze_folder(
    foldername)

if BATCHMSD:
    N = np.size(t) - 1
    np_batch = np.array(batch_msds)
    ave_batch = np.median(np_batch, axis=0)
    lower_quart = np.percentile(np_batch, 25, axis=0)
    upper_quart = np.percentile(np_batch, 75, axis=0)
    for k in range(len(batch_msds[0])):
        ave_batch[k] = np.average(batch_msds[:][k])
    plt.figure('All MSDs', dpi=300)
    for i in range(len(batch_msds)):
        plt.plot(t[0:N - 3], batch_msds[i][1:N], '-k', alpha=0.1, linewidth=0.5)  # a set of plain black lines shows the MSD for 3D movement for each of the analyzed trajectories
    # a red line shows the median MSD
    plt.plot(t[0:N - 3], ave_batch[1:N], '-r', alpha=1, linewidth=1.5)
    # the light red shading for the interquartile range
    plt.fill_between(t[0:N - 3], lower_quart[1:N], upper_quart[1:N], color='r', alpha=0.3, edgecolor='none')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time step (${\\tau}$), s', size=16)
    plt.ylabel('MSD, ${\mu}$m${^2}$/s', size=16)
    # plt.ylabel('MSD, a.u.', size=16)
    plt.plot([t[0], t[N - 3]], [ave_batch[1], ave_batch[1] * (N - 1)], 'k:')
    plt.subplots_adjust(bottom=0.15, top=0.95)
    plt.savefig(foldername + "/figures/" + "All MSDs.png", format="png")
    plt.show()
    

if BATCHMSD_XY:
    N = np.size(t) - 1
    np_batch_xy = np.array(batch_msds_xy)
    ave_batch_xy = np.median(np_batch_xy, axis=0)
    lower_quart_xy = np.percentile(np_batch_xy, 25, axis=0)
    upper_quart_xy = np.percentile(np_batch_xy, 75, axis=0)
    # for k in range(len(batch_msds[0])):
    #     ave_batch[k] = np.average(batch_msds[:][k])
    plt.figure('All MSDs in XY', dpi=300)
    for i in range(len(batch_msds_xy)):
        plt.plot(t[0:N - 3], batch_msds_xy[i][1:N], '-k', alpha=0.1,
                 linewidth=0.5)  # a plain black line shows the MSD for 3D movement
    plt.plot(t[0:N - 3], ave_batch_xy[1:N], '-r', alpha=1, linewidth=1.5)
    plt.fill_between(t[0:N - 3], lower_quart_xy[1:N], upper_quart_xy[1:N], color='r', alpha=0.3, edgecolor='none')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([t[0], t[N - 3]], [ave_batch_xy[1], ave_batch_xy[1] * (N - 1)], 'k:')
    plt.savefig(foldername + "/figures/" + "All MSDs in XY.png", format="png")
    plt.show()

if PRINTRESULTS:
    print("Results for the directory " + foldername)

    print("\nList of total displacements:")
    print(MSD3D_max)
    print("Average total displacement:")
    print(str(np.average(MSD3D_max)) + " +- " + str(np.std(MSD3D_max)))

    print("\nList of MSDs at dt=1:")
    print(MSD3D_0)
    print("Average of MSDs at dt=1:")
    print(str(np.average(MSD3D_0)) + " +- " + str(np.std(MSD3D_0)))

    print("\nList of D coefficients in 3D:")
    print(D3Ds)
    print("Average D:")
    print(str(np.average(D3Ds)) + " +- " + str(np.std(D3Ds)))

    print("\nList of D coefficients in X:")
    print(DXs)
    print("Average D:")
    print(str(np.average(DXs)) + " +- " + str(np.std(DXs)))

    print("\nList of D coefficients in Y:")
    print(DYs)
    print("Average D:")
    print(str(np.average(DYs)) + " +- " + str(np.std(DYs)))

    print("\nList of D coefficients in Z:")
    print(DZs)
    print("Average D:")
    print(str(np.average(DZs)) + " +- " + str(np.std(DZs)))

    print("\nList of median speeds:")
    print(speeds_median)
    print("Average speed:")
    print(str(np.average(speeds_median)) + " +- " + str(np.std(speeds_median)))

    print("\nList of total covered volumes:")
    print(volumes)
    print("Average covered volume:")
    print(str(np.average(volumes)) + " +- " + str(np.std(volumes)))

if SAVERESULTS:
    f = open((foldername + "results.txt"), "w+")
    f.write("Results for the directory " + foldername)

    f.write("\n\nList of total displacements:")
    f.write(str(MSD3D_max))
    f.write("\nMedian total displacement:")
    f.write(str(np.median(MSD3D_max)) + ", 25%: " + str(np.quantile(MSD3D_max, 0.25)) + ", 75%: " + str(
        np.quantile(MSD3D_max, 0.75)))

    f.write("\n\nList of MSDs at dt=1:")
    f.write(str(MSD3D_0))
    f.write("\nMedian of MSDs at dt=1:")
    f.write(str(np.median(MSD3D_0)) + ", 25%: " + str(np.quantile(MSD3D_0, 0.25)) + ", 75%: " + str(
        np.quantile(MSD3D_0, 0.75)))

    f.write("\n\nList of D coefficients in 3D:")
    f.write(str(D3Ds))
    f.write("\nMedian D3D:")
    f.write(str(np.median(D3Ds)) + ", 25%: " + str(np.quantile(D3Ds, 0.25)) + ", 75%: " + str(np.quantile(D3Ds, 0.75)))

    f.write("\n\nList of D coefficients in XY:")
    f.write(str(Dxys))
    f.write("\nMedian Dxy:")
    f.write(str(np.median(Dxys)) + ", 25%: " + str(np.quantile(Dxys, 0.25)) + ", 75%: " + str(np.quantile(Dxys, 0.75)))

    f.write("\n\nList of D coefficients in X:")
    f.write(str(DXs))
    f.write("\nMedian DX:")
    f.write(str(np.median(DXs)) + ", 25%: " + str(np.quantile(DXs, 0.25)) + ", 75%: " + str(np.quantile(DXs, 0.75)))

    f.write("\n\nList of D coefficients in Y:")
    f.write(str(DYs))
    f.write("\nMedian DY:")
    f.write(str(np.median(DYs)) + ", 25%: " + str(np.quantile(DYs, 0.25)) + ", 75%: " + str(np.quantile(DYs, 0.75)))

    f.write("\n\nList of D coefficients in Z:")
    f.write(str(DZs))
    f.write("\nMedian DZ:")
    f.write(str(np.median(DZs)) + ", 25%: " + str(np.quantile(DZs, 0.25)) + ", 75%: " + str(np.quantile(DZs, 0.75)))

    f.write("\n\nList of median speeds:")
    f.write(str(speeds_median))
    f.write("\nMedian speed:")
    f.write(str(np.median(speeds_median)) + ", 25%: " + str(np.quantile(speeds_median, 0.25)) + ", 75%: " + str(
        np.quantile(speeds_median, 0.75)))

    f.write("\n\nList of total covered volumes:")
    f.write(str(volumes))
    f.write("\nMedian covered volume:")
    f.write(str(np.median(volumes)) + ", 25%: " + str(np.quantile(volumes, 0.25)) + ", 75%: " + str(
        np.quantile(volumes, 0.75)))
    # f.write(str(np.average(volumes)) + " +- " + str(np.std(volumes)))
    f.close()
