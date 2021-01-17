import os
from collections import deque

import numpy as np
from gym import spaces, Env
import torch

# from .utils import source_bash_file
# source_bash_file(os.path.expanduser('~') + '/.bashrc')
# source_bash_file(os.path.expanduser('~') + '/ros/catkin_ws_modulation_rl/devel/setup.bash')
# source_bash_file(os.path.expanduser('~') + '/ros/catkin_ws_modulation_rl/devel/setup.sh')
from dynamic_system_py import PR2Env, TiagoEnv #, HSREnv


class ActionRanges:
    # ranges = {
    #     'base_rot': [-0.2, 0.2],
    #     'modulate_alpha_x': [-1.0, 1.0],
    #     'modulate_alpha_y': [-1.0, 1.0],
    #     'modulate_alpha_dir': [-np.pi/2, np.pi/2],
    #     'modulation_lambda1': [-2.0, 2.0],
    #     'pause_gripper': [0, 1],
    #     'base_x': [-0.02, 0.02],
    #     'base_y': [-0.02, 0.02],
    #     'tiago_base_vel': [-0.02, 0.02],
    #     'tiago_base_angle': [-0.04, 0.04],
    #     'gripper_x': [-0.01, 0.01],
    #     'gripper_y': [-0.01, 0.01],
    #     'gripper_z': [-0.01, 0.01],
    #     'gripper_dq_x': [-1.0, 1.0],
    #     'gripper_dq_y': [-1.0, 1.0],
    #     'gripper_dq_z': [-1.0, 1.0],
    #     'gripper_dq_w': [-1.0, 1.0],
    # }

    @classmethod
    def get_ranges(cls, env_name: str, strategy: str, pause_gripper: bool, arctan2_alpha: bool):
        ks = []
        if pause_gripper:
            ks.append('pause_gripper')

        if strategy == 'modulate':
            ks += ['base_rot', 'modulation_lambda1']
            ks += ['modulate_alpha_x', 'modulate_alpha_y'] if arctan2_alpha else ['modulate_alpha_dir']
        elif strategy in ['dirvel', 'relvelm', 'relveld', 'unmodulated']:
            if env_name == 'tiago':
                ks += ['tiago_base_vel', 'tiago_base_angle']
            else:
                ks += ['base_rot', 'base_x', 'base_y']

        n = len(ks)
        # min_actions = [cls.ranges[k][0] for k in ks]
        # max_actions = [cls.ranges[k][1] for k in ks]
        min_actions = n * [-1]
        max_actions = n * [1]

        return ks, min_actions, max_actions

# https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2019driveCVPRW.pdf
# https://github.com/utiasSTARS/bingham-rotation-learning
def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)

    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim, A_dim)
    A = A_vec.new_zeros((A_vec.shape[0], A_dim, A_dim))
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()

def A_vec_to_quat(A_vec):
    A = convert_Avec_to_A(A_vec)
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    _, evs = torch.symeig(A, eigenvectors=True)
    return evs[:, :, 0].squeeze()


class ModulationEnv(Env):
    """
    Environment for modulating the base of a mobile manipulator robot.

    Action space:
    - alpha_x    Direction of the norm vector in which to apply the velocity. Direction is set as arctan2(alpha_x, alpha_y)
    - alpha_y
    - lambda1    Velocity in direction of the norm vector. E.g. 1: no modulation, 2: double the velocity, -1: same speed in reverse direction
    - lambda2    Velocity orthogonal to norm vector. Not required if direction of the vector can be set. Input ignored and always set to 1 within the c++ code
    - lambda3    Angular velocity for the base
    """
    DoneReturnCode = {0: 'Ongoing',
                      1: 'Success: goal reached',
                      2: 'Failure: too many ik_fail or ik_fail at goal'}

    metadata = {'render.modes': []}

    def __init__(self,
                 env,
                 ik_fail_thresh: int,
                 ik_fail_thresh_eval: int,
                 penalty_scaling: float,
                 time_step: float,
                 slow_down_real_exec: float,
                 arctan2_alpha,
                 alpha_direct_rng,
                 seed: int,
                 use_base_goal: bool,
                 strategy: str,
                 real_execution: str,
                 init_controllers: bool,
                 pause_gripper_action: bool,
                 vis_env: bool,
                 transition_noise_ee: float,
                 transition_noise_base: float,
                 start_pause: float,
                 min_goal_dist: float = 1,
                 max_goal_dist: float = 5,
                 stack_k_obs: int = 1,
                 perform_collision_check: bool = False,
                 hsr_ik_slack_dist = None,
                 hsr_ik_slack_rot_dist: float = None,
                 hsr_sol_dist_reward: bool = None):
        """
        Args:
            start_pose_distribution: whether to start each episode from a random gripper pose
            ik_fail_thresh: number of kinetic failures after which to abort the episode
            ik_fail_thresh_eval: number of kinetic failures after which to abort the episode during evaluation (allows to compare across different ik_fail_thresh)
            penalty_scaling: how much to weight the penalty for large action modulations in the reward
            arctan2_alpha: whether to construct modulation_aplha as actan2 or directly learn it (in which case actions[1] will be ignored).
                Setting to false requires to also change the bounds for this action
            min_actions: lower bound constraints for actions
            max_actions: upper bound constraints for actions
        """
        assert 0 < min_goal_dist < max_goal_dist
        args = [seed,
                min_goal_dist,
                max_goal_dist,
                use_base_goal,
                strategy,
                real_execution,
                init_controllers,
                penalty_scaling,
                time_step,
                slow_down_real_exec,
                perform_collision_check
                ]
        if env == 'pr2':
            self._env = PR2Env(*args)
        elif env == 'tiago':
            self._env = TiagoEnv(*args)
        elif env == 'hsr':
            self._env = HSREnv(*args, hsr_ik_slack_dist, hsr_ik_slack_rot_dist, hsr_sol_dist_reward)
        else:
            raise ValueError('Unknown env')

        self.state_dim = self._env.get_obs_dim()
        print(f"Detected state dim: {self.state_dim}")

        ks, min_actions, max_actions = ActionRanges.get_ranges(env_name=env, strategy=strategy, pause_gripper=pause_gripper_action, arctan2_alpha=arctan2_alpha)
        print(f"Actions to learn: {ks}")
        self.action_dim = len(min_actions)
        self._min_actions = np.array(min_actions)
        self._max_actions = np.array(max_actions)
        self.action_names = ks

        # TODO: DEFINE OBS BOUNDS
        self.observation_space = spaces.Box(low=-100, high=100, shape=[stack_k_obs * self.state_dim])
        # NOTE: env does all the action scaling according to _min_actions and _max_actions
        # and just expects actions in the range [-1, 1] from the agent
        self.action_space = spaces.Box(low=np.array(self.action_dim * [-1.0]),
                                       high=np.array(self.action_dim * [1.0]),
                                       shape=[len(min_actions)])

        self._ik_fail_thresh = ik_fail_thresh
        self._ik_fail_thresh_eval = ik_fail_thresh_eval
        self._arctan2_alpha = arctan2_alpha
        self._alpha_direct_rng = alpha_direct_rng
        self._last_k_obs = deque(stack_k_obs * [np.zeros(self.state_dim)], maxlen=stack_k_obs)
        self._pause_gripper_action = pause_gripper_action
        self._strategy = strategy
        self._use_base_goal = use_base_goal
        self._env_name = env
        self._vis_env = vis_env
        self._transition_noise_ee = transition_noise_ee
        self._transition_noise_base = transition_noise_base
        self._start_pause = start_pause
        self.reset(start_pose_distribution="rnd", gripper_goal_distribution="rnd", success_thres_dist=0.02, success_thres_rot=0.05)

    def _parse_env_output(self, retval):
        obs = retval[:self.state_dim]
        reward = retval[self.state_dim]
        done_return = retval[self.state_dim + 1]
        nr_kin_failures = retval[self.state_dim + 2]
        return np.array(obs), reward, done_return, nr_kin_failures

    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: (np.ndarray) Action to scale
        :return: (np.ndarray) Scaled action
        """
        low, high = self._min_actions, self._max_actions
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self._min_actions, self._max_actions
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _convert_policy_to_env_actions(self, actions):
        # NOTE: ALL ACTION RANGES ARE [-1, 1] (ONLY modulation_alpha WE ADJUST WITHIN THIS FUNCTION). SO REGULARISE ACTIONS SIMILARLY WE CAN JUST SCARE EVERYTHIN
        # COULD PROBABLY REMOVE THE self.unscale_action(actions) CALL
        # stretch and translate actions from [-1, 1] range into target range
        actions = list(self.unscale_action(actions))

        if self._pause_gripper_action:
            # action is in range [-1, 1]
            pause_gripper = actions.pop(0) > 0.0
        else:
            pause_gripper = False

        # base actions
        if self._strategy == 'modulate':
            base_rot, lambda1 = actions.pop(0), actions.pop(0)
            if self._arctan2_alpha:
                modulation_alpha = np.arctan2(actions.pop(0), actions.pop(0))
            else:
                modulation_alpha = self._alpha_direct_rng * actions.pop(0)
            base_actions = [base_rot, lambda1, modulation_alpha]
        elif self._strategy in ['relvelm', 'relveld', 'dirvel', 'unmodulated']:
            if self._env_name == 'tiago':
                vel, angle = actions.pop(0), actions.pop(0)
                base_actions = [vel, angle]
            else:
                base_rot, base_x, base_y = actions.pop(0), actions.pop(0), actions.pop(0)
                base_actions = [base_rot, base_x, base_y]
        elif self._strategy == 'modulate_ellipse':
            base_actions = [0.0, 0.0, 0.0]
        else:
            raise ValueError(f"Unknown strategy {self._strategy}")

        assert len(actions) == 0, "Not all actions consumed"

        return pause_gripper, base_actions

    # needed for stable_baselines replay buffer class
    def normalize_reward(self, reward):
        return reward

    def _stack_obs(self, obs_list: list):
        return np.concatenate(obs_list, axis=0)

    @property
    def last_orig_obs(self):
        return self._stack_obs(self._last_k_obs)

    def reset(self,
              start_pose_distribution: str,
              gripper_goal_distribution: str,
              success_thres_dist: float,
              success_thres_rot: float,
              gripper_goal: list = None,
              base_start: list = None,
              close_gripper: bool = False,
              gmm_model_path: str = ""
              ):
        assert start_pose_distribution in ["fixed", "restricted_ws", "rnd"], start_pose_distribution
        assert (gripper_goal is not None) or (gripper_goal_distribution in ["fixed", "restricted_ws", "rnd"]), gripper_goal_distribution
        if gripper_goal_distribution is None:
            gripper_goal_distribution = ""
        if gripper_goal is None:
            gripper_goal = []
        if base_start is None:
            base_start = []

        if gmm_model_path:
            assert os.path.exists(gmm_model_path), f"Path {gmm_model_path} doesn't exist"

        if self.get_real_execution() == 'world':
            print("################################")
            print("Resetting episode")
            print("################################")
            print(f"Next gripper goal: {gripper_goal if gripper_goal else 'None'}, random start pose: {start_pose_distribution}, gmm model: {gmm_model_path if gmm_model_path else 'None'}")
            if gripper_goal:
                self.add_goal_marker(gripper_goal, 99999, "pink")
            if gripper_goal is None:
                input("dummy input")
                print("No gripper goal specified. Drawing location from possible parts of the map\n")
                while True:
                    user_input = input(f"Press y to accept or enter a new goal as [x y z R P Y] in world coordinates:\n")
                    if user_input == 'y':
                        continue
                    else:
                        user_goal = [float(g) for g in user_input.split(" ")]
                        if len(user_goal) == 6:
                            gripper_goal = user_goal
                            self.add_goal_marker(gripper_goal, 99999, "pink")
                            print(f"New specified gripper goal: {gripper_goal}")
                            continue
                        else:
                            f"Goal needs to have 6 values separated by white space. Received {gripper_goal}"

        orig_obs = self._env.reset(gripper_goal,
                                   base_start,
                                   start_pose_distribution,
                                   gripper_goal_distribution,
                                   close_gripper,
                                   gmm_model_path,
                                   success_thres_dist,
                                   success_thres_rot,
                                   self._start_pause,
                                   self._vis_env)

        for _ in range(len(self._last_k_obs) - 1):
            self._last_k_obs.appendleft(np.zeros(self.state_dim))
        self._last_k_obs.appendleft(orig_obs)

        return self._stack_obs(self._last_k_obs)

    def visualize(self, logdir: str = "", logfile: str = "") -> list:
        if logfile:
            os.makedirs(logdir, exist_ok=True)
            path = f'{logdir}/{logfile}'
        else:
            path = ""
        path_points = self._env.visualize(path)
        return path_points

    def step(self, action, eval=False):
        """
        Take a step in the environment.
        Args:
            action: array of length self.action_dim with values in range [-1, 1] (automatically scaled into correct range)
            eval: whether to use the ik_fail_thresh or ik_fail_thresh_eval

        Returns:
            obs: array of length self.state_dim
            reward (float): reward
            done_return (int): whether the episode terminated, see cls.DoneReturnCode
            nr_kin_failure (int): cumulative number of kinetic failures in this episode
        """
        thres = self._ik_fail_thresh_eval if eval else self._ik_fail_thresh

        pause_gripper, base_actions = self._convert_policy_to_env_actions(action)

        if eval:
            transition_noise_ee, transition_noise_base = 0, 0
        else:
            transition_noise_ee, transition_noise_base = self._transition_noise_ee, self._transition_noise_base

        retval = self._env.step(thres, pause_gripper, base_actions, transition_noise_ee, transition_noise_base)
        obs, reward, done_return, nr_kin_failures = self._parse_env_output(retval)

        # use unnormalised obs to have them for the replay buffer (see last_orig_obs())
        self._last_k_obs.appendleft(obs)

        stacked_obs = self._stack_obs(self._last_k_obs)

        info = {'nr_kin_failures': nr_kin_failures}

        return stacked_obs, reward, done_return, info

    def parse_done_return(self, code):
        """
        code (int): returned value from the env, integer in [0, 2]
        """
        return self.DoneReturnCode[code]

    def close(self):
        pass

    def get_obs(self):
        return self._env.get_obs()

    def get_dist_to_goal(self):
        """Return euclidean distance from current gripper position to gripper goal"""
        return self._env.get_dist_to_goal()

    def get_rot_dist_to_goal(self):
        """Return rotational distance from current gripper position to gripper goal"""
        return self._env.get_rot_dist_to_goal()

    def set_gripper_goal(self, goal: list, gripper_goal_distribution: str, gmm_model_path: str, success_thres_dist: float, success_thres_rot: float):
        """
        goal: [x, y, z, r, p, y] in world coordinates
        """
        if gmm_model_path:
            assert os.path.exists(gmm_model_path), f"Path {gmm_model_path} doesn't exist"
        if goal:
            assert gripper_goal_distribution is None
            gripper_goal_distribution = ""
        orig_obs = self._env.set_gripper_goal(goal, gripper_goal_distribution, gmm_model_path, success_thres_dist, success_thres_rot)
        for _ in range(len(self._last_k_obs) - 1):
            self._last_k_obs.appendleft(np.zeros(self.state_dim))
        self._last_k_obs.appendleft(orig_obs)

        return self._stack_obs(self._last_k_obs)

    def open_gripper(self, position: float = 0.08):
        if self.get_real_execution() == "sim":
            return True
        else:
            return self._env.open_gripper(position)

    def close_gripper(self, position: float = 0.00):
        if self.get_real_execution() == "sim":
            return True
        else:
            return self._env.close_gripper(position)

    @staticmethod
    def parse_obs(obs, precision=2, floatmode='fixed') -> dict:
        names = ["rel_gripper_pose_", 7, "planned_gripper_vel_rel_", 3, "planned_gripper_dq_", 4, "rel_gripper_goal", 7, "paused_count_", 1, "current_joint_values", -1]
        cum = 0
        parsed = dict()
        for i in range(len(names) // 2):
            n = names[(2*i) + 1]
            if n != -1:
                values = np.array(obs)[cum:cum+n]
            else:
                values = np.array(obs)[cum:]
            print(f"{names[2*i]}: {np.array2string(values, precision=precision, floatmode=floatmode)}")
            parsed[n] = values
            cum += n
        return parsed

    def add_goal_marker(self, pos: list, marker_id: int, color: str = "blue"):
        self._env.add_goal_marker(pos, marker_id, color)

    def set_real_execution(self, real_execution: str, time_step: float, slow_down_real_exec: float):
        self._env.set_real_execution(real_execution, time_step, slow_down_real_exec)

    def get_real_execution(self):
        return self._env.get_real_execution()

    def get_slow_down_factor(self):
        return self._env.get_slow_down_factor()

    def set_ik_slack(self, ik_slack_dist: float, ik_slack_rot_dist: float):
        assert self._env_name == 'hsr'
        self._env.set_ik_slack(ik_slack_dist, ik_slack_rot_dist)

    def clear(self):
        pass
