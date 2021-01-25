import os
from pathlib import Path
from typing import Callable, Union

import numpy as np
import wandb
from matplotlib import pyplot as plt

plt.style.use('seaborn')
import rospy

# from stable_baselines3.common.buffers import NStepReplayBuffer, ReplayBuffer
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList

from modulation.utils import setup_config
from modulation.stableBLCallbacks import ModulationEvalCallback, IKSlackScheduleCallback
from modulation.envs.env_utils import get_env, wrap_in_task


# do not upload model checkpoints
# os.environ['WANDB_IGNORE_GLOBS'] = '*.zip'


class UnmodulatedStableBLAgent:
    def __init__(self, env, gamma, tensorboard_log):
        self._env = env
        # just to make conform with stableBL agents
        self.num_timesteps = 0
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        raise ValueError("Unmodulated Agent cannot be saved or loaded")

    def get_env(self):
        return self._env

    def predict(self, obs, state=None, mask=None, deterministic=None):
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            shp = (len(self._env.get_attr('action_names')[0]))
        else:
            shp = (self._env.num_envs, len(self._env.get_attr('action_names')[0]))
        return np.zeros(shp), state


def sync_all(file_log, sync_model: bool):
    """
    sync all files so far to have them even if weird crash thingy happens
    not super robust, but cannot pass recursive argument to wandb
    """
    wandb.save(f'{file_log}/trajectories/*/*/*', base_path=file_log, policy='live')
    if sync_model:
        wandb.save(f'{file_log}/*', base_path=file_log, policy='live')
    wandb.log({}, commit=True)
    rospy.loginfo("File sync done")


def construct_agent(config, env, tensorboard_log, restore_model_path: str = None, restore_kwargs=None):
    if config.explore_noise:
        if config.explore_noise_type == 'normal':
            action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                             sigma=config.explore_noise * np.ones(env.action_space.shape))
        elif config.explore_noise_type == 'OU':
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape),
                                                        sigma=config.explore_noise * np.ones(env.action_space.shape))
        else:
            raise ValueError(f"Unknown action noise {config.explore_noise_type}")
    else:
        action_noise = None

    def create_lr_schedule(start_lr:float, min_lr: float, gamma: float, total_decay_steps: int) -> Union[float, Callable]:
        """
        Return an exponentially decaying lr. total_decay_steps maps from progrss to current step.
        Note: if using random exploration steps, the lr will already decay during those
        progress_remaining: falling from 1 to 0
        """

        # def exp_sched(progress_remaining):
        #     """decay per step -> way too fast for values like 0.99 -> would need to make it smth like per episode
        #     or add a decay_steps"""
        #     step = (1 - progress_remaining) * total_decay_steps / decay_steps
        #     return max(start_lr * gamma ** step, min_lr)

        if min_lr == -1:
            return start_lr

        def lin_sched(progress_remaining):
            return min_lr + progress_remaining * (start_lr - min_lr)

        assert 0 < gamma <= 1
        assert 0 < start_lr <= 0.1
        assert 0 <= min_lr <= start_lr
        assert total_decay_steps >= 1
        return lambda progress_remaining: lin_sched(progress_remaining)

    lr_fn = create_lr_schedule(start_lr=config.lr_start,
                               min_lr=config.lr_end,
                               gamma=config.lr_gamma,
                               total_decay_steps=1_000_000)

    common_args = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': lr_fn,
        'buffer_size': config.buffer_size,
        'learning_starts': config.rnd_steps,
        'batch_size': config.batch_size,
        'tau': config.tau,
        'gamma': config.gamma,
        'action_noise': action_noise,
        'tensorboard_log': tensorboard_log,
        'create_eval_env': False,
        'seed': config.seed,
        'verbose': 0,
        'device': config.device
    }

    rospy.loginfo(f"Constructing {config.algo} agent")
    if config.algo == 'SAC':
        agent = SAC(train_freq=1,
                    # gradient_steps=-1,
                    # n_episodes_rollout,
                    ent_coef=config.ent_coef,
                    target_update_interval=1,
                    target_entropy='auto',
                    use_sde=config.use_sde,
                    # sde_sample_freq=,
                    # use_sde_at_warmup=,
                    # policy_kwargs=
                    **common_args)
    elif config.algo == 'TD3':
        # nstep
        # pip install git+https://github.com/PartiallyTyped/stable-baselines3.git@nstep
        # if config.nstep != 1:
        #     replay_buffer_class = NStepReplayBuffer
        #     replay_buffer_kwargs = {'n_steps': config.nstep}
        # else:
        #     replay_buffer_class = None
        #     replay_buffer_kwargs = None
        agent = TD3(# DEFAULT SAC USES TRAIN_FREQ, DEFAULT TD3 N_EPISODES_ROLLOUT. But doesn't seem to make a difference for TD3 which one I use?
                    # train_freq=1,
                    # gradient_steps=-1,
                    n_episodes_rollout=1,
                    policy_delay=config.policy_frequency,
                    target_policy_noise=config.policy_noise,
                    target_noise_clip=config.noise_clip,
                    # replay_buffer_class=replay_buffer_class,
                    # replay_buffer_kwargs=replay_buffer_kwargs,
                    **common_args
                    )
    elif config.algo == 'unmodulated':
        return UnmodulatedStableBLAgent(env=env, gamma=config.gamma, tensorboard_log=tensorboard_log)
    else:
        raise NotImplementedError(f"Unknown algo {config.algo}")

    # this will overwrite the above constructed agents with the parameters loaded from the zipfile
    # TODO: replay buffer does not get stored and loaded yet
    if config.resume_id or restore_model_path:
        # in case we load a model other than the last the time-step won't match the last one logged and wandb will reject the logged values -> always set to last
        restore_kwargs = {"num_timesteps": wandb.run.step, "_total_timesteps": wandb.run.step}
        if config.resume_id:
            rospy.loginfo(f"Resuming existing agent from run {config.resume_id} with model {config.resume_model_name}")
            # downloads the file and stores it in the (new) wandb run dir
            model_file = wandb.restore(config.resume_model_name).name
            agent = agent.load(model_file, env, **restore_kwargs)
        elif restore_model_path:
            rospy.loginfo(f"Restoring existing agent from path {restore_model_path}")
            agent = agent.load(restore_model_path, env, **restore_kwargs)

    if config.debug and config.algo != 'unmodulated':
        try:
            wandb.watch(agent.actor, log='all', idx=0)
            wandb.watch(agent.critic, log='all', idx=1)
        except Exception as e:
            rospy.loginfo(f"Failed to watch gradients, might be due to already having constructed an agent before: {e}")
    return agent


def learning_loop(config, eval_env, agent, file_log: str):
    rospy.loginfo("Start learning loop")

    eval_callback = ModulationEvalCallback(eval_env=eval_env,
                                           n_eval_episodes=config.nr_evaluations,
                                           eval_freq=config.evaluation_frequency,
                                           log_path=file_log,
                                           best_model_save_path=file_log,
                                           debug=config.debug
                                           )
    if config.hsr_ik_slack_schedule:
        ik_slack_cb = IKSlackScheduleCallback(start_dist_slack=config.hsr_ik_slack_dist, end_dist_slack=0.02,
                                              start_rot_slack=config.hsr_ik_slack_rot_dist, end_rot_slack=0.05,
                                              total_timesteps=config.total_steps)
        cbs = CallbackList([eval_callback, ik_slack_cb])
    else:
        cbs = eval_callback
    agent.learn(total_timesteps=config.total_steps,
                callback=cbs,
                # log_interval=,
                eval_env=None,
                tb_log_name=config.algo,
                # reset_num_timesteps=
                )

    agent.save(os.path.join(eval_callback.best_model_save_path, f'last_model'))
    if not config.debug:
        sync_all(file_log, sync_model=True)
    rospy.loginfo("Training finished")


def evaluate_on_task(config, eval_env, agent, task: str, real_exec: str, file_log: str, time_step: float,
                     slow_down_real_exec: float, rm_task_objects: bool, n_evals: int = None):
    eval_env.env_method("set_real_execution", real_exec, time_step, slow_down_real_exec)

    task_eval_env = wrap_in_task(eval_env, task,
                                 start_launchfiles_no_controllers=config.start_launchfiles_no_controllers,
                                 rm_task_objects=rm_task_objects)
    rospy.loginfo(f"Evaluating on task {task_eval_env.get_attr('taskname')[0]} with {real_exec} execution.")

    prefix = 'final'
    if real_exec != 'sim':
        prefix += f'_ts{time_step}_slow{slow_down_real_exec}'

    eval_callback = ModulationEvalCallback(eval_env=task_eval_env,
                                           n_eval_episodes=n_evals or config.nr_evaluations,
                                           eval_freq=config.evaluation_frequency,
                                           log_path=file_log,
                                           best_model_save_path=None,
                                           debug=config.debug,
                                           prefix=prefix,
                                           verbose=2)
    eval_callback.init_callback(agent)
    # if in debug mode, set step to 0, so wandb will fail to log the values because step already exists
    eval_callback.num_timesteps = wandb.run.step + 5 if not config.debug else 0
    eval_callback.do_eval()

    # delete all spawned models again
    task_eval_env.env_method("clear")


# @traced
def main():
    # need a node to listen to some stuff for the task envs
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    main_path = Path(__file__).parent
    run, config = setup_config(main_path, sync_tensorboard=True)

    # USE SAME ENV FOR EVAL. OTHERWISE POTENTIAL NS CONFLICT WITH CONTROLLERS (PLUS NEED TO START ALL CONTROLLERS TWICE)
    env, eval_env = get_env(config, start_launchfiles=config.start_launchfiles_no_controllers, create_eval_env=False, task=config.task)

    tensorboard_log = f'{config.logpath}/stableBL/'
    file_log = f'{config.logpath}/mytmp/'
    agent = construct_agent(config, env, tensorboard_log,
                            restore_model_path=config.model_file if config.restore_model else None)

    # train
    if not config.evaluation_only:
        learning_loop(config, eval_env, agent, file_log=file_log)
        env.env_method("clear")
        eval_env.env_method("clear")

    # evaluate
    if (env.get_attr('_env_name') == 'hsr') and config.hsr_ik_slack_schedule:
        env.env_method('set_ik_slack', 0.02, 0.05)

    if not config.start_launchfiles_no_controllers:
        execs = ["world"] if (config.real_execution == "world") else config.eval_execs

        for real_exec in execs:
            for task in config.eval_tasks:
                if config.rm_task_objects is not None:
                    rm_task_objects = [config.rm_task_objects]
                elif (real_exec == 'gazebo') and (task in ['picknplace', 'door', 'drawer']):
                    rm_task_objects = [False, True]
                else:
                    rm_task_objects = [False]

                for rm in rm_task_objects:
                    evaluate_on_task(config, eval_env, agent=agent, task=task, real_exec=real_exec, file_log=file_log,
                                     time_step=config.time_step, slow_down_real_exec=config.slow_down_real_exec,
                                     rm_task_objects=rm)
                    if not config.debug:
                        sync_all(file_log, sync_model=False)

    rospy.signal_shutdown("We are done")


if __name__ == '__main__':
    main()
