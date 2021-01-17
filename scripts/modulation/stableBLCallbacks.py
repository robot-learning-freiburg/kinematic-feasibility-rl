import os
import time
import warnings
import numpy as np
from typing import Union, List, Dict, Any, Optional
import matplotlib.pyplot as plt
import gym
import torch

import rospy
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common import logger

from modulation.utils import EarlyStopperBestSaver, print_pathPoint
from modulation.visualise import plot_pathPoints, plot_relative_pose_map, plot_zfailure_hist

def calc_dist_to_sol(p):
    """NOTE: requires that planned_gripper_... is the plan that gripper_... tried to achieve"""
    return np.sqrt((p.gripper_x - p.planned_gripper_x) ** 2 +
                   (p.gripper_y - p.planned_gripper_y) ** 2 +
                   (p.gripper_z - p.planned_gripper_z) ** 2)


# from stable_baselines3.common.evaluation import evaluate_policy
def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    callback=None,
                    global_step=None, commit_logs=None,
                    plot_n_path_points=10,  # None to plot all
                    debug=False,
                    file_log_path: str = None,
                    main_prefix: str = 'eval',
                    verbose: int = 1):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    name_prefix = f"{main_prefix}_{env.get_attr('taskname')[0]}"
    spause = env.get_attr('_start_pause')[0]
    if spause:
        name_prefix += f'_spause{spause}'
    real_exec = env.env_method('get_real_execution')[0]
    if real_exec != "sim":
        name_prefix += f'/{real_exec}'
    n_rosbags = 10
    t = time.time()

    episode_rewards, episode_returns, episode_lengths, actions, kin_fails, pathPoints, dist_gripper_sols_success, dist_gripper_sols_fail, dist_gripper_sols_max, collisions, final_dist_to_goal = [], [], [], [], [], [], [], [], [], [], []
    ik_fail_thresh_eval = env.get_attr('_ik_fail_thresh_eval')[0]
    # slow_down = env.env_method('get_slow_down_factor')[0]
    max_len = 15_000
    with torch.no_grad():
        for i in range(n_eval_episodes):
            obs = env.env_method('reset')
            done, state = False, None
            episode_reward, episode_return, episode_length = 0.0, 0.0, 0
            episode_kin_fails, info = [], [{}]
            while not done:
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                pause_gripper, base_actions = env.env_method('_convert_policy_to_env_actions', action[0])[0]
                actions += [[pause_gripper] + base_actions]

                if debug:
                    assert not np.isnan([[pause_gripper] + base_actions]).sum(), "Nan found in actions"

                # dummy vec env automatically calls reset as soon as done, which also resets the tracetories we want to visualise
                # also allows me to pass in eval=True
                obs, reward, done, info = env.env_method('step', action=action[0], eval=True)[0]
                obs, reward, done, info = [obs], [reward], done, [info]

                if episode_length >= 10_000:
                    if episode_length % 10_000 == 0:
                        rospy.logwarn(f"{episode_length} steps already!. Continuing until a max. of {max_len}")
                    if episode_length > max_len:
                        pathPoint = env.env_method('visualize')[0]
                        assert episode_length < max_len, f"EPISODE OF {episode_length} STEPS!"

                episode_reward += reward[0]
                episode_return += (model.gamma ** episode_length) * reward[0]
                episode_kin_fails.append(info[0]['nr_kin_failures'])

                if debug:
                    assert not np.isnan(obs).sum(), "Nan found in obs"
                    assert (model.gamma ** episode_length) * reward[0] <= 0.001, (reward[0], (model.gamma ** episode_length) * reward[0])
                    assert reward[0] <= 0.001

                if callback is not None:
                    callback(locals(), globals())
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            # kin fails are cumulative
            kin_fails.append(episode_kin_fails[-1])
            final_dist_to_goal.append(env.env_method('get_dist_to_goal')[0])

            if i in np.linspace(0, n_eval_episodes, n_rosbags, dtype=np.int):
                log_dir, logfile = f'{file_log_path}/trajectories/{global_step}/{name_prefix}', f'e{i}'
            else:
                log_dir, logfile = "", ""
            pathPoint = env.env_method('visualize', log_dir, logfile)[0]
            collisions.append(np.sum([p.collision for p in pathPoint]))

            dists_to_sol = [calc_dist_to_sol(p) for p in pathPoint]
            dist_gripper_sols_max.append(max(dists_to_sol))
            dist_gripper_sols_success.append(np.mean([dists_to_sol[i] for i, p in enumerate(pathPoint) if not p.ik_fail]))
            if episode_kin_fails[-1]:  # nan if there are no failures
                dist_gripper_sols_fail.append(np.mean([dists_to_sol[i] for i, p in enumerate(pathPoint) if p.ik_fail]))
            pathPoints.append(pathPoint)

            if (verbose > 1) or (real_exec != "sim"):
                rospy.loginfo(f"{name_prefix}: Eval ep {i}: {(time.time() - t) / 60:.2f} minutes. "
                              f"Ik failures: {kin_fails[-1]}, collisions: {collisions[-1]}, steps: {episode_length}. "
                              f"{(np.array(kin_fails) == 0).sum()}/{i + 1} zero failure.")
                t = time.time()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if dist_gripper_sols_success:
        wandb.log({f"{name_prefix}/dist2gripperSol_success": np.mean(dist_gripper_sols_success)}, step=global_step)
    if dist_gripper_sols_fail:
        wandb.log({f"{name_prefix}/dist2gripperSol_fail": np.mean(dist_gripper_sols_fail)}, step=global_step)
    wandb.log({f"{name_prefix}/dist2gripperSol_below0.1": (np.array(dist_gripper_sols_max) < 0.1).mean()}, step=global_step)
    wandb.log({f"{name_prefix}/dist2gripperSol_below0.05": (np.array(dist_gripper_sols_max) < 0.05).mean()}, step=global_step)
    wandb.log({f"{name_prefix}/dist2gripperSol_below0.1_plusReached": ((np.array(final_dist_to_goal) <= 0.1) * (np.array(dist_gripper_sols_max) < 0.1)).mean()}, step=global_step)
    wandb.log({f"{name_prefix}/dist2gripperSol_below0.05_plusReached": ((np.array(final_dist_to_goal) <= 0.05) * (np.array(dist_gripper_sols_max) < 0.05)).mean()}, step=global_step)
    # wandb.log({f"{name_prefix}/dist2gripperSol_max": dist_gripper_sols_max}, step=global_step)
    wandb.log({f"{name_prefix}/dist2gripperSol_hist": wandb.Histogram(np_histogram=np.histogram(dist_gripper_sols_max,
                                                                                                    bins=min(len(dist_gripper_sols_max), 50),
                                                                                                    density=True,
                                                                                                    range=(0, 1)))}, step=global_step)

    # wandb.log({f"{name_prefix}/collisions": np.mean(collisions)}, step=global_step)
    # wandb.log({f"{name_prefix}/collisions_zero": (np.array(collisions) == 0).sum() / n_eval_episodes}, step=global_step)
    # wandb.log({f'{name_prefix}/collisions_hist': wandb.Histogram(np_histogram=np.histogram(collisions, bins=64, density=True))}, step=global_step)

    fig_pathPoints = plot_pathPoints(pathPoints[:plot_n_path_points] if plot_n_path_points else pathPoints)
    wandb.log({f"{name_prefix}/pathPoints": wandb.Image(fig_pathPoints)}, step=global_step)

    fig_relPose2d, fig_relPose3d = plot_relative_pose_map(pathPoints)
    wandb.log({f"{name_prefix}/relPose2D": wandb.Image(fig_relPose2d)}, step=global_step)
    wandb.log({f"{name_prefix}/relPose3D": wandb.Image(fig_relPose3d)}, step=global_step)

    f_zfailure_hist = plot_zfailure_hist(pathPoints)
    wandb.log({f"{name_prefix}/zfailure_hist": wandb.Image(f_zfailure_hist)}, step=global_step)

    # if eval_goals is not None:
    #     f = env.env_method('visualise_goals', kin_fails)
    #     wandb.log({f"{name_prefix}/goalImg": wandb.Image(f)}, step=global_step)

    fails_per_episode = np.array(kin_fails)
    final_dist_to_goal = np.array(final_dist_to_goal)
    wandb.log({f'{name_prefix}/kin_fails_hist': wandb.Histogram(np_histogram=np.histogram(fails_per_episode, bins=64, density=True, range=(0, ik_fail_thresh_eval)))}, commit=None, step=global_step)
    actions = np.array(actions)
    for a, name in enumerate(env.get_attr('action_names')[0]):
        # 'modulate' strategy learns one more actions than are passed to the env
        if name == 'modulate_alpha_x':
            name = 'modulation_alpha'
        if name == 'modulate_alpha_y':
            continue
        wandb.log({f'actions_{name_prefix}/{name}': wandb.Histogram(np_histogram=np.histogram(actions[:, a], bins=64, density=True))},
                  commit=None, step=global_step)

    metrics = {f'return_undisc':        sum(episode_rewards) / n_eval_episodes,
               f'return_disc':          sum(episode_returns) / n_eval_episodes,
               f'epoch_len':            sum(episode_lengths) / n_eval_episodes,
               f'ik_below_eval_thres':  (fails_per_episode <= ik_fail_thresh_eval).sum() / n_eval_episodes,
               f'ik_below_11':          (fails_per_episode < 11).sum() / n_eval_episodes,
               f'ik_zero_fail':         (fails_per_episode == 0).sum() / n_eval_episodes,
               f'num_kin_fail':         fails_per_episode.sum() / n_eval_episodes,
               f'goal_reached':         (final_dist_to_goal <= env.get_attr('_success_thres_dist')[0]).sum() / n_eval_episodes,
               'total_timesteps':       global_step,
               'global_step':           global_step
               }
    rospy.loginfo("---------------------------------------")
    rospy.loginfo("T {:}, {:} evaluation over {:.0f} episodes: Avg. return (undisc) {:.3f}, Avg. return (disc) {:.3f}, Avg failures {:.2f}".format(global_step, name_prefix, n_eval_episodes, metrics[f'return_undisc'], metrics[f'return_disc'], metrics[f'num_kin_fail']))
    rospy.loginfo("IK fails: {:.3f}p < {:}, {:.3f}p < 11, {:.3f}p < 1".format(metrics[f'ik_below_eval_thres'], ik_fail_thresh_eval, metrics[f'ik_below_11'], metrics[f'ik_zero_fail']))
    rospy.loginfo("---------------------------------------")
    wandb.log({(f'{name_prefix}/{k}' if ('step' not in k) else k): v for k, v in metrics.items()}, commit=None, step=global_step)

    plt.close('all')
    return episode_rewards, episode_lengths, metrics, name_prefix


class ModulationEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: (bool) Whether to render or not the environment during evaluation
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 early_stop_metric: str = None,
                 comparison: str = None,
                 early_stop_after_evals: int = None,
                 n_avg: int = None,
                 debug: bool = False,
                 prefix: str = 'eval'
                 ):
        super(ModulationEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path

        self.earlyStopperBestSaver = EarlyStopperBestSaver(early_stop_metric,
                                                           comparison,
                                                           early_stop_after_evals,
                                                           n_avg)

        self.debug = debug
        self.prefix = prefix

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type"
                          f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def do_eval(self):
        # Sync training and eval env if there is VecNormalize
        sync_envs_normalization(self.training_env, self.eval_env)
        return evaluate_policy(self.model, self.eval_env,
                               n_eval_episodes=self.n_eval_episodes,
                               deterministic=self.deterministic,
                               global_step=self.num_timesteps,
                               commit_logs=None,
                               debug=self.debug,
                               file_log_path=self.log_path,
                               main_prefix=self.prefix,
                               verbose=self.verbose)
    
    def _on_step(self) -> bool:
        continue_train = True

        if self.n_calls and (self.n_calls % 100000 == 0) and (self.best_model_save_path is not None):
            self.model.save(os.path.join(self.best_model_save_path, f'model_t{self.num_timesteps}'))

        if (self.n_calls == 1) or (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            episode_rewards, episode_lengths, metrics, name_prefix = self.do_eval()

            # only using to adapt the wandb summary values so far
            # not using do_early_stop so far, could return ~do_early to stop
            is_better, _do_early_stop = self.earlyStopperBestSaver.process_new_metric(metrics, name_prefix)
            # continue_train = not _do_early_stop

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            if self.verbose > 0:
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            if is_better:
                if self.best_model_save_path is not None:
                    print("Saving best model")
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return continue_train


class IKSlackScheduleCallback(EventCallback):
    def __init__(self, start_dist_slack: float, end_dist_slack: float, start_rot_slack: float, end_rot_slack: float, total_timesteps: int):
        super(IKSlackScheduleCallback, self).__init__()
        self._start_dist_slack = start_dist_slack
        self._end_dist_slack = end_dist_slack
        self._start_rot_slack = start_rot_slack
        self._end_rot_slack = end_rot_slack

        self._total_timesteps = total_timesteps
    def _on_step(self) -> bool:
        if self.training_env.get_attr('_env_name')[0] == 'hsr':
            progress_remaining = (self._total_timesteps - self.num_timesteps) / self._total_timesteps

            def lin_sched(progress_remaining, start, end):
                return end + progress_remaining * (start - end)

            self.training_env.env_method('set_ik_slack', lin_sched(progress_remaining, self._start_dist_slack, self._end_dist_slack),
                                                         lin_sched(progress_remaining, self._start_rot_slack, self._end_rot_slack))
        return True