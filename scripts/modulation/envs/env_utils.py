import time
from typing import Tuple, Optional, Any
from gym import Wrapper

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

from modulation.envs.tasks import RndStartRndGoalsTask, RestrictedWsTask, BaseChainedTask, PickNPlaceChainedTask, DoorChainedTask, DrawerChainedTask
from modulation.envs.modulationEnv import ModulationEnv
from modulation.handle_launchfiles import start_launch_files, stop_launch_files


def wrap_in_task(env, task: str, start_launchfiles_no_controllers: bool, rm_task_objects: bool):
    if isinstance(env, ModulationEnv):
        orig_env = env
    else:
        orig_env = env.envs[0]
        while not isinstance(orig_env, ModulationEnv):
            orig_env = orig_env.env

    assert isinstance(orig_env, ModulationEnv), orig_env
    task = task.lower()
    if task == "picknplace":
        task_env = PickNPlaceChainedTask(orig_env, rm_task_objects)
    elif task == 'door':
        task_env = DoorChainedTask(orig_env, rm_task_objects)
    elif task == 'drawer':
        task_env = DrawerChainedTask(orig_env, rm_task_objects)
    elif task == 'restrictedws':
        task_env = RestrictedWsTask(orig_env)
    elif task == 'rndstartrndgoal':
        task_env = RndStartRndGoalsTask(orig_env)
    else:
        raise NotImplementedError(f"Task {task} wrapper not added yet")

    if isinstance(task, BaseChainedTask):
        assert not start_launchfiles_no_controllers, "We need gazebo to spawn objects etc"

    return DummyVecEnv([lambda: task_env])


def get_env(config,
            start_launchfiles: bool = True,
            create_eval_env: bool = False,
            task: str = None) -> Tuple[Any, Any]:
    """
    Startup gazebo to load robot config, create env and close gazebo back down.
    Shutting it down might display a bunch of error messages
    """

    """
    from subprocess import check_output, call
    from pathlib import Path
    import os
    my_env = os.environ.copy()
    my_env['PATH'] = ":".join([p for p in my_env['PATH'].split(":") if "conda" not in p])
    my_env['PYTHONPATH'] = ":".join([p for p in my_env['PYTHONPATH'].split(":") if "catkin_ws_modulation_rl_py3" not in p])
    python_interpreter = check_output("which python2".split(" ")).decode('utf-8').strip()
    cmd = [python_interpreter, str(Path(__file__).parent.parent / 'handle_launchfiles.py'), "--env", config.env]
    if task:
        cmd += ['--use_task_world']
    success = call(cmd, env=my_env)
    """


    if start_launchfiles:
        print("WARNING: IF STARTING NODES FROM PYTHON CONTROLLERS WILL FAIL DUE TO WRONG ROS ENV (NOT PY2.7). DON'T USE WITH TASKS OR REAL EXECUTION")
        assert config.real_execution == "sim", config.real_execution
        # a = subprocess.run('which python2'.split(" "), stdout=subprocess.PIPE)
        # python2_path = a.stdout.strip()
        # p_gazebo, p_moveit = subprocess.run([python2_path, Path(__file__).parent / 'handle_launchfiles.py', '--env', config.env], stdout=subprocess.PIPE)
        p_gazebo, p_moveit = start_launch_files(config.env)
    else:
        p_gazebo, p_moveit = None, None

    def _create_env(config, task: str) -> ModulationEnv:
        env = ModulationEnv(env=config.env,
                            ik_fail_thresh=config.ik_fail_thresh,
                            ik_fail_thresh_eval=config.ik_fail_thresh_eval,
                            penalty_scaling=config.penalty_scaling,
                            time_step=config.time_step,
                            slow_down_real_exec=config.slow_down_real_exec,
                            seed=config.seed,
                            strategy=config.strategy,
                            real_execution=config.real_execution,
                            init_controllers=not config.start_launchfiles_no_controllers,
                            stack_k_obs=config.stack_k_obs,
                            perform_collision_check=config.perform_collision_check,
                            vis_env=config.vis_env,
                            transition_noise_ee=config.transition_noise_ee,
                            transition_noise_base=config.transition_noise_base,
                            start_pause=config.start_pause,
                            hsr_ik_slack_dist=config.hsr_ik_slack_dist,
                            hsr_ik_slack_rot_dist=config.hsr_ik_slack_rot_dist,
                            hsr_sol_dist_reward=config.hsr_sol_dist_reward)
        return wrap_in_task(env=env, task=task,
                            start_launchfiles_no_controllers=config.start_launchfiles_no_controllers,
                            rm_task_objects=config.rm_task_objects)

    print(f"Creating {config.env}")
    env = _create_env(config, task=task)
    if create_eval_env:
        eval_env = _create_env(config, task=task)
    else:
        eval_env = env

    if start_launchfiles:
        time.sleep(10)
        # let moveit continue to run as we sometimes need the planning scene for the collision avoidance
        stop_launch_files(p_gazebo, p_moveit)

    return env, eval_env


# class MaxStepWrapper(Wrapper):
#     def __init__(self, env, max_episode_steps):
#         super(MaxStepWrapper, self).__init__(env)
#         self._max_episode_steps = max_episode_steps
#         self._elapsed_steps = None
#
#     def __getattr__(self, name):
#         # if name.startswith('_'):
#         #     raise AttributeError("attempted to get missing private attribute '{}'".format(name))
#         return getattr(self.env, name)
#
#     def step(self, **kwargs):
#         assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
#         observation, reward, done, info = self.env.step(**kwargs)
#         self._elapsed_steps += 1
#         if self._elapsed_steps >= self._max_episode_steps:
#             info['TimeLimit.truncated'] = not done
#             done = True
#         return observation, reward, done, info
#
#     def reset(self, **kwargs):
#         self._elapsed_steps = 0
#         return self.env.reset(**kwargs)