import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from modulation.utils import rpy_to_quiver_uvw


def plot_pathPoints(path_points: list, show_planned_gripper: bool = True, show_actual_gripper: bool = True,
                    show_base: bool = True):
    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j, path_point in enumerate(path_points):
        base_x, base_y, base_rot, gripper_x, gripper_y, planned_gripper_x, planned_gripper_y, ik_fails = 8 * [
            np.array([])]
        for i, p in enumerate(path_point):
            base_x = np.append(base_x, p["base_x"])
            base_y = np.append(base_y, p["base_y"])
            base_rot = np.append(base_rot, p["base_rot"])
            gripper_x = np.append(gripper_x, p["gripper_x"])
            gripper_y = np.append(gripper_y, p["gripper_y"])
            planned_gripper_x = np.append(planned_gripper_x, p["planned_gripper_x"])
            planned_gripper_y = np.append(planned_gripper_y, p["planned_gripper_y"])
            ik_fails = np.append(ik_fails, p["ik_fail"])
        # only plot every xth point
        idx = np.arange(1, len(path_point), 10)

        c = ccycle[j % len(ccycle)]
        if show_planned_gripper:
            ax.plot(planned_gripper_x, planned_gripper_y, ls=':', color=c, label=None)
            ax.scatter(planned_gripper_x[ik_fails.astype(np.bool)], planned_gripper_y[ik_fails.astype(np.bool)],
                       color='r', marker='X', linewidths=0.001)
        if show_actual_gripper:
            ax.plot(gripper_x, gripper_y, ls='--', color=c, label=None)
        if show_base:
            ax.quiver(base_x[idx], base_y[idx], 1.0 * np.ones_like(base_rot[idx]), 1.0 * np.ones_like(base_rot[idx]),
                      angles=np.rad2deg(base_rot[idx]),
                      scale_units='dots', scale=0.09, width=0.002, headwidth=4, headlength=2, headaxislength=2,
                      pivot='middle',
                      color=c, label=f'{j}')
    mapsize = 6
    ax.set_ylim([-mapsize, mapsize])
    ax.set_xlim([-mapsize, mapsize])
    ax.legend()
    ax.set_title("Base and gripper paths")
    return f


def plot_relative_pose_map(path_points: list, max_path_points=25, max_points=4_000):
    path_points = path_points[:max_path_points]
    gripper_rel_x, gripper_rel_y, gripper_rel_z, gripper_rel_R, gripper_rel_P, gripper_rel_Y, ik_fails = 7 * [np.array([])]

    # subsample to a max number of points
    total_points = sum([len(p) for p in path_points])
    nth = max(1, np.ceil(total_points / max_points))
    counter = 0

    for j, path_point in enumerate(path_points):
        for i, p in enumerate(path_point):
            if (counter % nth) == 0:
                gripper_rel_x = np.append(gripper_rel_x, p["gripper_rel_x"])
                gripper_rel_y = np.append(gripper_rel_y, p["gripper_rel_y"])
                gripper_rel_z = np.append(gripper_rel_z, p["gripper_rel_z"])
                gripper_rel_R = np.append(gripper_rel_R, p["gripper_rel_R"])
                gripper_rel_P = np.append(gripper_rel_P, p["gripper_rel_P"])
                gripper_rel_Y = np.append(gripper_rel_Y, p["gripper_rel_Y"])
                ik_fails = np.append(ik_fails, p["ik_fail"])
            counter += 1

    f, ax = plt.subplots(1, 1, figsize=(14, 12))
    # sns.histplot(x=gripper_rel_x, y=gripper_rel_y, bins=50, cmap="mako", cbar=True, stat='probability', ax=ax)
    sns.scatterplot(x=gripper_rel_x, y=gripper_rel_y, hue=ik_fails, cmap="mako", ax=ax)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_title("Relative gripper poses top-down")
    # plt.show(); plt.close()

    f3d, ax3d = plt.subplots(1, 1, figsize=(14, 12))
    ax3d = f3d.add_subplot(111, projection='3d')
    n = Normalize(vmin=0, vmax=1, clip=True)
    cmap = plt.get_cmap('coolwarm')
    cs = cmap(n(ik_fails))
    U, V, W = rpy_to_quiver_uvw(roll=gripper_rel_R, pitch=gripper_rel_P, yaw=gripper_rel_Y)
    ax3d.quiver(gripper_rel_x, gripper_rel_y, gripper_rel_z, U, V, W,
                color=cs, edgecolor=cs, facecolor=cs, normalize=True, length=0.03, arrow_length_ratio=0.5, alpha=0.4)
    ax3d.set_title("Relative gripper poses")
    # plt.show(); plt.close()

    return f, f3d


def plot_zfailure_hist(path_points: list):
    # ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gripper_rel_z, ik_fails = 2 * [np.array([])]
    for j, path_point in enumerate(path_points):
        for i, p in enumerate(path_point):
            # only take every 10th point, assuming the pose won't be changing radically in one step
            if i % 20:
                gripper_rel_z = np.append(gripper_rel_z, p["gripper_rel_z"])
                ik_fails = np.append(ik_fails, p["ik_fail"])

    f, ax = plt.subplots(1, 1, figsize=(14, 12))
    # sns.histplot(x=gripper_rel_x, y=gripper_rel_y, bins=50, cmap="mako", cbar=True, stat='probability', ax=ax)
    sns.histplot(x=gripper_rel_z, hue=ik_fails, multiple='stack', ax=ax)
    ax.set_title("Kinematic failures per height of the gripper")
    return f
