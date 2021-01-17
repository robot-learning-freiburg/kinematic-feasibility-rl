import time
from typing import List
from pathlib import Path
from collections import namedtuple
import random
import numpy as np
from gym import Wrapper
import copy
from enum import IntEnum
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import rospy
from geometry_msgs.msg import Pose, Quaternion, Point
from modulation.envs.myGazebo import MyGazebo, GazeboObject
from modulation.envs.modulationEnv import ModulationEnv
from modulation.utils import rpy_to_quiver_uvw


class GripperActions(IntEnum):
    NONE = 0
    OPEN = 1
    GRASP = 2

TaskGoal = namedtuple("TaskGoal", ['pose', 'motion_plan', 'end_action', 'success_thres_dist', 'success_thres_rot'])


def ask_user_goal(goal_name: str, default_value: list, use_quaternion_rot: bool):
    if use_quaternion_rot :
        fmt = "[x y z X Y Z W]"
        goal_len = 7
    else:
        fmt = "[x y z R P Y]"
        goal_len = 6

    while True:
        if default_value:
            prompt = f"Last specified {goal_name}: {default_value}" \
                     f"Press a to accept or enter a new goal as {fmt} in world coordinates:\n"
        else:
            prompt = f"Enter a new {goal_name} as {fmt} in world coordinates:\n"
        user_input = input(prompt)
        if default_value and (user_input == 'a'):
            return default_value
        else:
            try:
                user_goal = [float(g) for g in user_input.split(" ")]
                if len(user_goal) == goal_len:
                    rospy.loginfo(f"Received {goal_name}: {user_goal}")
                    return user_goal
            except:
                rospy.loginfo(f"Invalid input: goal needs to have {goal_len} values separated by white space. Received {user_input}")


class WorldObjects:
    """
    name in gazebo database, x, y, z dimensions
    see e.g. sdfs here: https://github.com/osrf/gazebo_models/blob/master/table/model.sdf
    """
    # TODO: correct sizes for all
    coke_can = GazeboObject('coke_can', 0.05, 0.05, 0.1)
    table = GazeboObject("table", 1.5, 0.8, 1.0)
    kitchen_table = GazeboObject("kitchen_table", 0.68, 1.13, 0.68)
    # our own, non-database objects
    muesli2 = GazeboObject('muesli2', 0.05, 0.15, 0.23)
    kallax2 = GazeboObject('Kallax2', 0.415, 0.39, 0.65)
    kallax = GazeboObject('Kallax', 0.415, 0.39, 0.65)
    kallaxDrawer1 = GazeboObject('KallaxDrawer1', 0.415, 0.39, 0.65)
    # ATM CRASHING GAZEBO
    tim_bowl = GazeboObject('tim_bowl', 0.05, 0.05, 0.1)
    reemc_table_low = GazeboObject('reemc_table_low', 0.75, 0.75, 0.41)


class BaseTask(Wrapper):
    @property
    def taskname(self) -> str:
        raise NotImplementedError()

    def __getattr__(self, name):
        # if name.startswith('_'):
        #     raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def __init__(self,
                 env: ModulationEnv,
                 start_pose_distribution: str,
                 gripper_goal_distribution: str,
                 success_thres_dist: float = 0.025,
                 success_thres_rot: float = 0.05
                 ):
        self._success_thres_dist = success_thres_dist
        self._success_thres_rot = success_thres_rot
        self._start_pose_distribution = start_pose_distribution
        self._gripper_goal_distribution = gripper_goal_distribution
        super(BaseTask, self).__init__(env=env)

    def reset(self, gripper_goal=None, base_start=None, close_gripper=False, gmm_model_path=""):
        return self.env.reset(start_pose_distribution=self._start_pose_distribution,
                              gripper_goal_distribution=self._gripper_goal_distribution,
                              success_thres_dist=self._success_thres_dist,
                              success_thres_rot=self._success_thres_rot,
                              gripper_goal=gripper_goal,
                              base_start=base_start,
                              close_gripper=close_gripper,
                              gmm_model_path=gmm_model_path)

    def step(self, action, eval=False):
        return self.env.step(action=action, eval=eval)

    def clear(self):
        self.env.clear()


class RndStartRndGoalsTask(BaseTask):
    @property
    def taskname(self) -> str:
        return "rStart_rGoals"

    def __init__(self, env: ModulationEnv):
        super(RndStartRndGoalsTask, self).__init__(env=env,
                                                   start_pose_distribution='rnd',
                                                   gripper_goal_distribution='rnd')


class RestrictedWsTask(BaseTask):
    @property
    def taskname(self) -> str:
        return "restrictedWs"

    def __init__(self, env: ModulationEnv):
        super(RestrictedWsTask, self).__init__(env=env,
                                               start_pose_distribution='restricted_ws',
                                               gripper_goal_distribution='restricted_ws')


class RndStartFixedGoalsTask(BaseTask):
    # TODO: HOW TO MAKE THE EVAL-CALLBACK EVALUATE OVER ALL THE GOALS AND NOT JUST --nr_evaluations?
    @property
    def taskname(self) -> str:
        return "rStart_rGoals"

    def __init__(self, env: ModulationEnv):
        super(RndStartFixedGoalsTask, self).__init__(env=env,
                                                     start_pose_distribution="rnd",
                                                     gripper_goal_distribution="rnd")
        self.eval_goals = self.get_eval_goals()
        self._last_goal = 0

    def get_eval_goals(self, dist: float = 3.0, n: int = 10, z: float = 0.6):
        """Create gripper goals in a circle around the starting position with 6 rotatitions each"""
        rots = [[0, 0, 0],
                [0, 0.5, 0],
                [0, 1.0, 0],
                [0, 1.5, 0],
                [0, 0, 0.5],
                [0, 0, 1.5], ]
        rots = np.array([np.pi * np.array(r) for r in rots])  # [6, 3]

        angles = np.linspace(0, 2 * np.pi, n)
        xs = dist * np.cos(angles)
        ys = dist * np.sin(angles)
        zs = z * np.ones_like(ys)

        locs = np.stack([xs, ys, zs], axis=1)  # [n, 3]

        locs = np.tile(locs, [6, 1])  # [6 * n, 3]
        rots = np.repeat(rots, n, axis=0)  # [6 * n, 3]
        goals = np.concatenate([locs, rots], axis=1)  # [6 * n, 6]
        return goals

    def _get_next_goal(self):
        next_goal = self.eval_goals
        self._last_goal = (self._last_goal + 1) % len(self.eval_goals)
        return next_goal

    def reset(self, gripper_goal: list = None, start_pose_distribition: str = None, gripper_goal_distribution: str = None, base_start: list = None):
        return super(RndStartFixedGoalsTask, self).reset(gripper_goal=self._get_next_goal())

    def visualise_goals(self, kin_fails):
        f = plt.figure(figsize=(12, 6))
        ax = f.add_subplot(111, projection='3d')
        n = Normalize(vmin=0, vmax=20, clip=True)
        cmap = plt.get_cmap('plasma')

        # https://stackoverflow.com/questions/1568568/how-to-convert-euler-angles-to-directional-vector
        U, V, W = rpy_to_quiver_uvw(roll=self.eval_goals[:, 3], pitch=self.eval_goals[:, 4], yaw=self.eval_goals[:, 5])
        ax.quiver(self.eval_goals[:, 0], self.eval_goals[:, 1], self.eval_goals[:, 2], U, V, W,
                  color=cmap(n(kin_fails)), normalize=True, length=0.1, arrow_length_ratio=0.0)
        ax.set_zlim([0, 2])
        f.colorbar(cm.ScalarMappable(norm=n, cmap=cmap))
        return f


class BaseChainedTask(BaseTask):
    GRIPPER_CLOSED_AT_START = True

    def __init__(self, env: ModulationEnv, without_objects: bool):
        super(BaseChainedTask, self).__init__(env=env,
                                              start_pose_distribution="rnd",
                                              gripper_goal_distribution="rnd")

        self._motion_model_path = Path(__file__).parent.parent.parent.parent / "GMM_models"
        assert self._motion_model_path.exists(), self._motion_model_path

        if env.get_real_execution() == "world":
            self._gazebo = None
        else:
            self._gazebo: MyGazebo = MyGazebo()

        self._current_goal = 0
        # will be set at every reset
        self._goals = []
        self._without_objects = without_objects

    def spawn_scene(self) -> List[TaskGoal]:
        raise NotImplementedError()

    def grasp(self) -> List[TaskGoal]:
        raise NotImplementedError()

    def reset(self, gripper_goal: list = None, start_pose_distribition: str = None, gripper_goal_distribution: str = None, base_start: list = None):
        # first reset, then spawn scene & set actual goal to ensure we don't spawn objects into robot
        super(BaseChainedTask, self).reset(base_start=self.BASE_START_RNG,
                                           close_gripper=self.GRIPPER_CLOSED_AT_START)
        if not self.GRIPPER_CLOSED_AT_START:
            self.env.open_gripper()

        self._current_goal = 0
        self._gazebo.pause_physics()
        self._goals = self.spawn_scene()
        if self._without_objects:
            self._gazebo.delete_all_spawned()
        self._gazebo.unpause_physics()
        first_goal = self._goals[self._current_goal]

        return self.env.set_gripper_goal(goal=first_goal.pose, gripper_goal_distribution=None, gmm_model_path=first_goal.motion_plan,
                                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot)

    def _episode_cleanup(self):
        self.env.open_gripper()
        self._gazebo.pause_physics()
        self._gazebo.delete_all_spawned()
        self._gazebo.unpause_physics()

    def step(self, action, eval=False):
        normed_stacked_obs, reward, done_return, info = self.env.step(action=action, eval=eval)

        interactive = self.env.get_real_execution() == "world"
        if done_return == 1:
            end_action = self._goals[self._current_goal].end_action
            if interactive:
                input(f"Goal {self._current_goal} reached. Enter to {end_action} and start next goal.")
            if end_action == GripperActions.GRASP:
                self.grasp()
            elif end_action == GripperActions.OPEN:
                self.env.open_gripper()

            if self._current_goal < len(self._goals) - 1:
                new = self._goals[self._current_goal + 1]
                self.env.set_gripper_goal(goal=new.pose, gripper_goal_distribution=None, gmm_model_path=new.motion_plan,
                                          success_thres_dist=new.success_thres_dist, success_thres_rot=new.success_thres_rot)
                self._current_goal += 1
                done_return = 0

        # ensure nothing left attached to the robot / the robot could spawn into / ...
        if done_return:
            self._episode_cleanup()

        return normed_stacked_obs, reward, done_return, info

    def clear(self):
        if self._gazebo:
            self._gazebo.clear()
        super(BaseChainedTask, self).clear()


class PickNPlaceChainedTask(BaseChainedTask):
    # [min_x, max_x, min_y, max_y, min_yaw, max_yaw (radians)]
    BASE_START_RNG = [-1.5, 1.5, -1.5, 1.5, -0.5*np.pi, 0.5*np.pi]
    START_TABLE_POS = Point(x=3, y=0, z=0)
    END_TABLE_RNG = [-1.5, 2, -3, -2.5]

    @property
    def taskname(self) -> str:
        name = "picknplace"
        if (self.env.get_real_execution() != 'sim') and self._without_objects:
            name += 'NoObj'
        return name

    def __init__(self, env: ModulationEnv, without_objects: bool):
        super(PickNPlaceChainedTask, self).__init__(env=env, without_objects=without_objects)

        self._pick_obj = WorldObjects.muesli2
        self._pick_table = WorldObjects.reemc_table_low
        self._place_table = WorldObjects.reemc_table_low

        # real world helpers
        self._rw_last_object_loc = None
        self._rw_place_loc = None

        self.reset()

    def spawn_scene(self) -> List[TaskGoal]:
        """
        Robot spawns at a random location and rotation around the center (BASE_START_RNG).
        Fixed start table with object on a random point on the front edge.
        Random goal table position on the other end of the room (END_TABLE_RNG).
        """
        if self.env.get_real_execution() == "world":
            input("Dummy input")
            obj_loc = ask_user_goal(goal_name='object location', default_value=self._rw_last_object_loc, use_quaternion_rot=False)
            self._rw_last_object_loc = obj_loc

            in_front_of_obj_loc = copy.deepcopy(obj_loc)
            in_front_of_obj_loc[0] -= 0.2
            rospy.loginfo(f"Generated goal in front of object: {in_front_of_obj_loc}\n")

            place_loc = ask_user_goal(goal_name='PLACED object location', default_value=self._rw_place_loc, use_quaternion_rot=False)
            self._rw_place_loc = place_loc

            # NOTE: assumes place loc is 90 degree & to the right of the origin
            in_front_of_place_loc = copy.deepcopy(place_loc)
            in_front_of_place_loc[1] += 0.2

            goals = [TaskGoal(pose=in_front_of_obj_loc, motion_plan="", end_action=GripperActions.OPEN, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot),
                     TaskGoal(pose=obj_loc, motion_plan="", end_action=GripperActions.GRASP, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot),
                     TaskGoal(pose=in_front_of_place_loc, motion_plan="", end_action=GripperActions.NONE, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot),
                     TaskGoal(pose=place_loc, motion_plan="", end_action=GripperActions.OPEN, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot)]
        else:
            self._gazebo.delete_all_spawned()
            # self._gazebo.reset_world()

            start_table_pose = Pose(self.START_TABLE_POS, Quaternion(0, 0, 1, 0))
            self._gazebo.spawn_model("pick_table", self._pick_table, start_table_pose)

            endx = random.uniform(self.END_TABLE_RNG[0], self.END_TABLE_RNG[1])
            endy = random.uniform(self.END_TABLE_RNG[2], self.END_TABLE_RNG[3])
            end_table_pose = Pose(Point(x=endx, y=endy, z=0), Quaternion(0, 0, 1, 1))
            # self._gazebo.set_model("place_table", end_table_pose)
            self._gazebo.spawn_model("place_table", self._place_table, end_table_pose)

            # place target on edge of the table (relative to the table position)
            # NOTE: won't be correct yet if table not in front of robot
            x = self._pick_table.x / 2 - 0.1
            y = random.uniform(-self._pick_table.y + 0.1, self._pick_table.y - 0.1) / 2
            z = self._pick_table.z + self._pick_obj.z + 0.01
            pose_on_table = Pose(Point(x=x, y=y, z=z), Quaternion(0, 0, 1, 1))
            self._gazebo.spawn_model("pick_obj", self._pick_obj, pose_on_table, "pick_table::link")

            # to get the target position
            self._gazebo.spawn_model("place_obj", self._pick_obj, pose_on_table, "place_table::link")
            time.sleep(1)

            def get_grip_from_above_goals() -> list:
                raise NotImplementedError("TODO: update goal definitions")
                # derive gripper goals as global # [x, y, z, R, P, Y]
                height_offset = 0.1

                world_target_pos = self._gazebo.get_model("pick_table", "world").pose.position
                rot = [0, 0.5 * np.pi, 0]
                above_target    = [world_target_pos.x, world_target_pos.y, world_target_pos.z + height_offset] + rot
                at_target       = [world_target_pos.x, world_target_pos.y, world_target_pos.z] + rot

                # spawn model in gazebo relative to end table to get relative and global position (there might be an easier way)
                # self._gazebo.spawn_model("place_obj", self._pick_obj, pose_on_table, "place_table::link")
                world_end_target_pos = self._gazebo.get_model("place_obj", "world").pose.position
                rot = [0, 0.5 * np.pi, -np.pi / 2]
                above_end_table = [world_end_target_pos.x, world_end_target_pos.y, world_end_target_pos.z + height_offset] + rot
                at_end          = [world_end_target_pos.x, world_end_target_pos.y, world_end_target_pos.z] + rot

                # goals = [TaskGoal(pose=in_front_of_obj_loc, motion_plan="", end_action=GripperActions.OPEN),
                #          TaskGoal(pose=obj_loc, motion_plan="", end_action=GripperActions.GRASP),
                #          TaskGoal(pose=in_front_of_place_loc, motion_plan="", end_action=GripperActions.NONE),
                #          TaskGoal(pose=place_loc, motion_plan="", end_action=GripperActions.OPEN)]
                return goals

            def get_grip_from_front_goals() -> list:
                # NOTE: 'in_front' goals assume current rotation of tables relative to robot
                # pick goals
                world_target_pos = self._gazebo.get_model("pick_obj", "world").pose.position
                obj_loc = [world_target_pos.x, world_target_pos.y, world_target_pos.z - 0.04] + [0, 0, 0]
                in_front_of_obj_loc = copy.deepcopy(obj_loc)
                in_front_of_obj_loc[0] -= 0.2

                # place goals
                world_end_target_pos = self._gazebo.get_model("place_obj", "world").pose.position
                self._gazebo.delete_model("place_obj")
                place_loc = [world_end_target_pos.x, world_end_target_pos.y + 0.05, world_end_target_pos.z + 0.05] + [0, 0, -np.pi / 2]
                in_front_of_place_loc = copy.deepcopy(place_loc)
                in_front_of_place_loc[1] += 0.2

                return [TaskGoal(pose=in_front_of_obj_loc, motion_plan="", end_action=GripperActions.OPEN, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot),
                        TaskGoal(pose=obj_loc, motion_plan="", end_action=GripperActions.GRASP, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot),
                        TaskGoal(pose=in_front_of_place_loc, motion_plan="", end_action=GripperActions.NONE, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot),
                        TaskGoal(pose=place_loc, motion_plan="", end_action=GripperActions.OPEN, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot)]

            # goals = get_grip_from_above_goals()
            goals = get_grip_from_front_goals()

        for i, g in enumerate(goals):
            self.env.add_goal_marker(g.pose, 9999 + i)

        return goals

    def grasp(self):
        if self.env.get_real_execution() == "world":
            self.env.close_gripper(self._pick_obj.x - 0.02)
        else:
            self.env.close_gripper(0.0)


class DoorChainedTask(BaseChainedTask):
    GRIPPER_CLOSED_AT_START = False
    BASE_START_RNG = [-1.5, 1.5, -1.5, 1.5, -0.0 * np.pi, 1.0 * np.pi]
    SHELF_POS = Point(x=0.0, y=3.0, z=0.24)
    # DOOR_ANGLE_OPEN_RNG = [np.pi / 4, np.pi / 2]

    @property
    def taskname(self) -> str:
        name = "door"
        if (self.env.get_real_execution() != 'sim') and self._without_objects:
            name += 'NoObj'
        return name

    def __init__(self, env: ModulationEnv, without_objects: bool):
        super(DoorChainedTask, self).__init__(env=env, without_objects=without_objects)

        self._shelf = WorldObjects.kallax2

        # helpers for real world
        self._rw_last_door_goal = []

        self.reset()

    def spawn_scene(self) -> List[TaskGoal]:
        """
        Robot spawns at a random location and rotation around the center (BASE_START_RNG).
        Two shelfs stacked on top of each other spawn within SHELF_RNG.
        First goal set at the handle of the shelf.
        Second goal?
        """
        if self.env.get_real_execution() == "world":
            input("dummy input")
            gripper_goal = ask_user_goal(goal_name='CLOSED door origin', default_value=self._rw_last_door_goal, use_quaternion_rot=True)
            self._rw_last_door_goal = gripper_goal
            grasp_goal = TaskGoal(pose=gripper_goal, motion_plan=str(self._motion_model_path / "GMM_grasp_KallaxTuer.csv"), end_action=GripperActions.GRASP,
                                  success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot)
        else:
            self._gazebo.delete_all_spawned()

            self._gazebo.spawn_model("Kallax2_bottom", self._shelf, Pose(self.SHELF_POS, Quaternion(0, 0, 0, 1)))
            p = copy.deepcopy(self.SHELF_POS)
            p.z = 0.65
            self._gazebo.spawn_model("target_shelf", self._shelf, Pose(p, Quaternion(0, 0, 0, 1)))
            time.sleep(1)

            # grasp goal
            self._gazebo.set_joint_angle(model_name="target_shelf", joint_names=['/DoorJoint'], angles=[0])
            self._gazebo.unpause_physics()
            door_pose_closed = self._gazebo.get_link_state("target_shelf::Door")
            grasp_goal = TaskGoal(pose=[door_pose_closed.position.x, door_pose_closed.position.y + 0.01, door_pose_closed.position.z,
                                        door_pose_closed.orientation.x, door_pose_closed.orientation.y, door_pose_closed.orientation.z, door_pose_closed.orientation.w],
                                  motion_plan=str(self._motion_model_path / "GMM_grasp_KallaxTuer.csv"),
                                  end_action=GripperActions.GRASP,
                                  success_thres_dist=self._success_thres_dist,
                                  success_thres_rot=self._success_thres_rot)

        # opening goal: expects the object pose in the beginning of the movement
        # angle = random.uniform(self.DOOR_ANGLE_OPEN_RNG[0], self.DOOR_ANGLE_OPEN_RNG[1])
        # self.set_door_angle("shelf2", angle)
        # door_pose_release = self._gazebo.get_link_state("shelf2::Door")
        opening_goal = TaskGoal(pose=grasp_goal.pose,
                                motion_plan=str(self._motion_model_path / "GMM_move_KallaxTuer.csv"),
                                end_action=GripperActions.OPEN,
                                success_thres_dist=self._success_thres_dist,
                                success_thres_rot=self._success_thres_rot)

        # release goal: expects the object pose in the beginning of the movement -> leave undefined here and add after completing the open movement
        # for goals that start at pose 0
        # release_goal = DoorGoal(pose=None,
        #                         motion_plan=str(self._motion_model_path / "GMM_release_KallaxTuer.csv"))

        goals = [grasp_goal, opening_goal] #, release_goal]

        # for i, g in enumerate(goals):
        #     self.env.add_goal_marker(g.pose, 9999 + i)

        return goals

    def grasp(self):
        if self.env.get_real_execution() == "world":
            self.env.close_gripper(0.005)
        else:
            self.env.close_gripper(0.0)


class DrawerChainedTask(BaseChainedTask):
    GRIPPER_CLOSED_AT_START = False
    BASE_START_RNG = [-1.5, 1.5, -1.5, 1.5, 0.5 * np.pi, 1.5 * np.pi]
    DRAWER_POS = Point(x=-3.0, y=0.0, z=0.24)

    @property
    def taskname(self) -> str:
        name = "drawer"
        if (self.env.get_real_execution() != 'sim') and self._without_objects:
            name += 'NoObj'
        return name

    def __init__(self, env: ModulationEnv, without_objects: bool):
        super(DrawerChainedTask, self).__init__(env=env, without_objects=without_objects)

        self._drawer = WorldObjects.kallax
        # helpers for real world
        self._rw_last_drawer_goal = []

        self.reset()

    def spawn_scene(self) -> List[TaskGoal]:
        """
        Robot spawns at a random location and rotation around the center (BASE_START_RNG).
        Two shelfs stacked on top of each other spawn within SHELF_RNG.
        First goal set at the handle of the shelf.
        Second goal?
        """
        if self.env.get_real_execution() == "world":
            input("dummy input")
            gripper_goal = ask_user_goal(goal_name='CLOSED drawer origin', default_value=self._rw_last_drawer_goal, use_quaternion_rot=True)
            self._rw_last_drawer_goal = gripper_goal
            grasp_goal = TaskGoal(pose=gripper_goal, motion_plan=str(self._motion_model_path / "GMM_grasp_KallaxDrawer.csv"), end_action=GripperActions.GRASP, success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot)
        else:
            self._gazebo.delete_all_spawned()

            self._gazebo.spawn_model("Kallax_bottom", self._drawer, Pose(self.DRAWER_POS, Quaternion(0, 0, 1, 1)))
            p = copy.deepcopy(self.DRAWER_POS)
            p.z = 0.65
            self._gazebo.spawn_model("target_drawer", self._drawer, Pose(p, Quaternion(0, 0, 1, 1)))
            time.sleep(1)

            # grasp goal
            # self._gazebo.set_joint_angle(model_name="target_drawer", joint_names=['/Drawer1Joint'], angles=[0])
            self._gazebo.unpause_physics()
            door_pose_closed = self._gazebo.get_link_state("target_drawer::Drawer1")
            grasp_goal = TaskGoal(pose=[door_pose_closed.position.x + 0.04, door_pose_closed.position.y, door_pose_closed.position.z + 0.05,
                                        0, 0, 0, 1],
                                  motion_plan=str(self._motion_model_path / "GMM_grasp_KallaxDrawer.csv"),
                                  end_action=GripperActions.GRASP,
                                  success_thres_dist=self._success_thres_dist,
                                  success_thres_rot=self._success_thres_rot)

        opening_goal = TaskGoal(pose=grasp_goal.pose,
                                motion_plan=str(self._motion_model_path / "GMM_move_KallaxDrawer.csv"),
                                end_action=GripperActions.OPEN,
                                success_thres_dist=self._success_thres_dist,
                                success_thres_rot=self._success_thres_rot)

        goals = [grasp_goal, opening_goal] #, release_goal]

        # for i, g in enumerate(goals):
        #     self.env.add_goal_marker(g.pose, 9999 + i)

        return goals

    def grasp(self):
        if self.env.get_real_execution() == "world":
            self.env.close_gripper(0.005)
        else:
            self.env.close_gripper(0.0)
