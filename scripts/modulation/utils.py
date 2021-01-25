import argparse
import collections
import copy
import os
import random
import shutil
import sys
from typing import List

import numpy as np
import rospy
import torch
import wandb
import yaml
from matplotlib import pyplot as plt

from modulation.envs.modulationEnv import ModulationEnv


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_config(config_path, tags: List = None, sync_tensorboard=False, version: float = 8.3):
    if tags is None:
        tags = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_best_defaults', type=str2bool, nargs='?', const=True, default=False, help="Replace default values with those from configs/best_defaults.yaml.")
    parser.add_argument('--seed', type=int, default=-1, help="Set to a value >= to use deterministic seed")
    parser.add_argument('--rnd_steps', type=int, default=0, help='Number of random actions to record before starting with rl, subtracted from total_steps')
    parser.add_argument('--total_steps', type=int, default=1_000_000, help='Total number of action/observation steps to take over all episodes')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    #################################################
    # ALGORITHMS
    #################################################
    parser.add_argument('--algo', type=str, default='SAC', choices=['TD3', 'SAC', 'unmodulated'])
    parser.add_argument('--gamma', type=float, default=0.99, help='discount')
    parser.add_argument('--lr_start', type=float, default=1e-5)
    parser.add_argument('--lr_end', type=float, default=1e-6, help="Final / min learning rate. -1 to not decay")
    parser.add_argument('--lr_gamma', type=float, default=0.999, help='adam decay')
    parser.add_argument('--tau', type=float, default=0.001, help='target value moving average speed')
    parser.add_argument('--explore_noise_type', type=str, default='normal', choices=['normal', 'OU', ''], help='Type of exploration noise')
    parser.add_argument('--explore_noise', type=float, default=0.75, help='')
    parser.add_argument('--nstep', type=int, default=1, help='Use nstep returns. Currently only correct TD3. Requires a specific stable-baselines branch.')
    #################################################
    # TD3
    #################################################
    parser.add_argument('--policy_noise', type=float, default=0.1, help='noise added to target policy in critic update')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='range to clip target policy noise')
    parser.add_argument('--policy_frequency', type=int, default=2, help='Frequency with which the policy will be updated')
    #################################################
    # SAC
    #################################################
    parser.add_argument('--use_sde', type=str2bool, nargs='?', const=True, default=False, help="use sde exploration instead of action noise. Automatically sets explore_noise_type to None")
    parser.add_argument('--ent_coef', default="auto", help="Entropy coefficient. 'auto' to learn it.")
    #################################################
    # Env
    #################################################
    parser.add_argument('--env', type=str.lower, default='pr2', choices=['pr2', 'pr2old', 'tiago', 'hsr'], help='')
    parser.add_argument('--task', type=str.lower, default='rndstartrndgoal', choices=['rndstartrndgoal', 'restrictedws', 'picknplace', 'door', 'drawer', 'rndstartrndspline'], help='Train on a specific task env. Might override some other choices.')
    parser.add_argument('--rm_task_objects', type=str2bool, nargs='?', const=True, default=None, help='Delete the physical objects in the gazebo simulator before executing the motions.')
    parser.add_argument('--time_step', type=float, default=0.02, help='Time steps at which the RL agent makes decisions. Action repeat to scale it up to 50hz during real execun')
    parser.add_argument('--slow_down_real_exec', type=float, default=1.0, help='How much to slow down the planned gripper trajectories during real execun')
    parser.add_argument('--real_execution', type=str, default="sim", choices=["sim", "gazebo", "world"], help="What kind of movement execution and where to get updated values from. Sim: analytical environemt, don't call controllers, gazebo: gazebo simulator, world: real world")
    # parser.add_argument('--norm_obs', type=str2bool, nargs='?', const=True, default=False, help='')
    parser.add_argument('--stack_k_obs', type=int, default=1, help='number of (past) obs to stack and return as current obs. 1 to just return current obs')
    parser.add_argument('--strategy', type=str.lower, default="dirvel", choices=["relvelm", "relveld", "dirvel", "modulate_ellipse", "unmodulated"], help='What velocities to learn: modulate, velocity relative to the gripper velocity, direct base velocity')
    parser.add_argument('--ik_fail_thresh', type=int, default=19, help='number of failures after which on it is considered as failed (i.e. failed: failures > ik_fail_thresh)')
    parser.add_argument('--ik_fail_thresh_eval', type=int, default=99, help='different eval threshold to make comparable across settings and investigate if it can recover from failures')
    parser.add_argument('--penalty_scaling', type=float, default=0.01, help='by how much to scale the penalties to incentivise minimal modulation')
    parser.add_argument('--perform_collision_check', type=str2bool, nargs='?', const=True, default=True, help='Use the planning scen to perform collision checks (both with environment and self collisions)')
    parser.add_argument('--vis_env', type=str2bool, nargs='?', const=True, default=False, help='Whether to publish markers to rviz')
    parser.add_argument('--transition_noise_ee', type=float, default=0.0, help='Std of Gaussian noise applied to the next gripper transform during training')
    parser.add_argument('--transition_noise_base', type=float, default=0.0, help='Std of Gaussian noise applied to the next base transform during training')
    parser.add_argument('--start_pause', type=float, default=0.0, help='Seconds to wait before starting the EE-motion (allowing the base to position itself)')
    #################################################
    # HSR
    #################################################
    parser.add_argument('--hsr_ik_slack_dist', type=float, default=0.1, help='Allowed slack for the ik solution')
    parser.add_argument('--hsr_ik_slack_rot_dist', type=float, default=0.05, help='Allowed slack for the ik solution')
    parser.add_argument('--hsr_ik_slack_schedule', type=str2bool, default=False, help='Whether to linearly reduce the ik_slack down to the success_thresh')
    parser.add_argument('--hsr_sol_dist_reward', type=str2bool, default=False, help='Penalise distance to perfect ik solution')
    #################################################
    # Eval
    #################################################
    parser.add_argument('--nr_evaluations', type=int, default=50, help='Nr of runs for the evaluation')
    parser.add_argument('--evaluation_frequency', type=int, default=20000, help='In nr of steps')
    parser.add_argument('--evaluation_only', type=str2bool, nargs='?', const=True, default=False, help='If True only model will be loaded and evaluated no training')
    parser.add_argument('--eval_execs', nargs='+', default=['gazebo', 'sim'], help='Eval execs to run')
    parser.add_argument('--eval_tasks', nargs='+', default=['rndstartrndgoal', 'restrictedws', 'picknplace', 'door', 'drawer'], choices=['rndstartrndgoal', 'restrictedws', 'picknplace', 'door', 'drawer'], help='Eval tasks to run')
    #################################################
    # wandbstuff
    #################################################
    parser.add_argument('--resume_id', type=str, default=None, help='wandb id to resume')
    parser.add_argument('--resume_model_name', type=str, default='last_model.zip', help='If specifying a resume_id, which model to restore')
    parser.add_argument('--restore_model', type=str2bool, nargs='?', const=True, default=False, help='Restore the model and config saved in /scripts/model_checkpoints/${env}/.')
    parser.add_argument('--name', type=str, default="", help='wandb display name for this run')
    parser.add_argument('--name_suffix', type=str, default="", help='suffix for the wandb name')
    parser.add_argument('--tags', type=str, nargs='+', default=[], help='wandb tags')
    parser.add_argument('--group', type=str, default=None, help='wandb group')
    parser.add_argument('--use_name_as_group', type=str2bool, nargs='?', const=True, default=True, help='use the name as group')
    parser.add_argument('--project_name', type=str, default='rl_modulation', help='wandb project name')
    parser.add_argument('-d', '--dry_run', type=str2bool, nargs='?', const=True, default=False, help='whether not to log this run to wandb')
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help='log gradients to wandb, potentially extra verbosity')
    parser.add_argument('--start_launchfiles_no_controllers', type=str2bool, nargs='?', const=True, default=False, help='start launchfiles through training script. Means controllers will fail due to wrong python version. -> only use for pure training w/o tasks or real execution')

    args = parser.parse_args()
    args = vars(args)

    # user-specified command-line arguments
    cl_args = [k.replace('-', '').split('=')[0] for k in sys.argv]

    if args.pop('load_best_defaults'):
        with open(config_path / 'configs' / 'best_defaults.yaml') as f:
            best_defaults = yaml.safe_load(f)
        # replace with best_default value unless something else was specified through command line
        for k, v in best_defaults[args['env']].items():
            if k not in cl_args:
                args[k] = v

    # consistency checks for certain arguments
    if args['strategy'] in ['modulate_ellipse', 'unmodulated']:
        assert args['algo'] == 'unmodulated'
        assert args['evaluation_only']
        if args['strategy'] == 'modulate_ellipse':
            assert args['env'] == 'pr2', args['env']
    if args['algo'] == 'unmodulated':
        assert args['strategy'] in ['modulate_ellipse', 'unmodulated']
    if args['use_sde']:
        print("Using sde, setting explore_noise_type to None")
        args['explore_noise_type'] = None
    if args['nstep'] != 1:
        raise NotImplementedError("Currently need to use the nstep branch of the stableBL repo to have this supported")
        assert args['nstep'] > 0
        assert args['algo'] == 'TD3', "Not correctly implemented for SAC yet"
    if args['resume_id'] or args['restore_model']:
        assert args['evaluation_only'], "Continuing to train not supported atm (replay buffer doesn't get saved)"
    if args['env'] == 'hsr' and args['perform_collision_check']:
        print("SETTING perform_collision_check TO FALSE FOR HSR (RISK OF CRASHING GAZEBO)")
        args['perform_collision_check'] = False
    if args['env'] == 'hsr':
        assert not args['perform_collision_check'], "Collisions seem to potentially crash due to some unsupported geometries"

    n = args.pop('name')
    group = args.pop('group')
    use_name_as_group = args.pop('use_name_as_group')
    if not n:
        n = []
        for k, v in sorted(args.items()):
            if (v != parser.get_default(k)) and (k not in ['env', 'seed', 'load_best_defaults', 'name_suffix', 'version',
                                                           'start_launchfiles_no_controllers', 'evaluation_only', 'vis_env',
                                                           'resume_id', 'eval_tasks', 'eval_execs', 'total_steps', 'perform_collision_check']):
                n.append(str(v) if (type(v) == str) else f'{k}:{v}')
        n = '_'.join(n)
    run_name = '_'.join([j for j in [args['env'], n, args.pop('name_suffix')] if j])

    if use_name_as_group:
        assert not group, "Don't specify a group and use_name_as_group"
        rname = run_name[:99] if len(run_name) > 99 else run_name
        group = rname + f'_v{version}'

    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['logpath'] = f'{config_path}/logs'
    os.makedirs(args['logpath'], exist_ok=True)
    args['version'] = version

    if args['dry_run']:
        os.environ['WANDB_MODE'] = 'dryrun'

    common_args = {'project': args.pop('project_name'),
                   'dir': args['logpath'],
                   'tags': [f'v{version:.1f}'] + tags + args.pop('tags'),
                   'sync_tensorboard': sync_tensorboard,
                   'group': group}
    if args['resume_id']:
        assert not args['dry_run']
        run = wandb.init(id=args['resume_id'],
                         resume=args['resume_id'],
                         **common_args)
    elif args['restore_model']:
        p = config_path / 'model_checkpoints' / args['env']
        print(f"RESTORING MODEL FOR {args['env']} from {p}")

        with open(p / 'config.yaml') as f:
            raw_params = yaml.safe_load(f)
        params = {k: v['value'] for k, v in raw_params.items() if k not in ['_wandb', 'wandb_version']}

        params['model_file'] = p / args['resume_model_name']
        params['restore_model'] = True
        params['resume_id'] = None

        run = wandb.init(config=params, **common_args)
        if args['evaluation_only']:
            wandb.config.update({"evaluation_only": True}, allow_val_change=True)
    else:
        # delete all past wandb runs. Mostly relevant for running sweeps in docker which might fill up the space o/w
        delete_dir(os.path.join(common_args['dir'], 'wandb'))
        delete_dir(os.path.join(common_args['dir'], 'mytmp'))
        run = wandb.init(config=args, name=run_name, **common_args)

    if args['resume_id'] or args['restore_model']:
        # update an alternative dict placeholder so we don't change the logged values which it was trained with
        config = DotDict(copy.deepcopy(dict(wandb.config)))

        for k, v in args.items():
            # allow to override loaded config with command line args
            if k in cl_args:
                config[k] = v
            # backwards compatibility if a config value didn't exist before
            if k not in wandb.config.keys():
                print(f"Key {k} not found in config. Setting to {v}")
                config[k] = args[k]
        # always update these values
        for k in ['start_launchfiles_no_controllers', 'device']:
            config[k] = args[k]
    else:
        config = wandb.config

    if config.evaluation_only:
        if not (config.resume_id or config.restore_model or (config.algo == 'unmodulated')):
            print("Evaluation only but no model to load specified! Evaluating a randomly initialised agent.")

    print(f"Log path: {config.logpath}")

    # Set seeds
    # NOTE: if changing the args wandb will not get the change in sweeps as they don't work over the command line!!!
    if config.seed == -1:
        wandb.config.update({"seed": random.randint(10, 1000)}, allow_val_change=True)
        config['seed'] = wandb.config.seed

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    return run, config


class DotDict(dict):
    """
    Source: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    Example:
    m = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def delete_dir(dirname: str):
    try:
        print(f"Deleting dir {dirname}")
        shutil.rmtree(dirname)
    except Exception as e:
        print(f"Failed to delete dir {dirname}: {e}")


def print_pathPoint(pathPoint):
    print(f"base_x: {pathPoint.base_x:.3f}")
    print(f"base_y: {pathPoint.base_y:.3f}")
    print(f"base_rot: {pathPoint.base_rot:.3f}")
    print(f"gripper_x: {pathPoint.gripper_x:.3f}")
    print(f"gripper_y: {pathPoint.gripper_y:.3f}")
    print(f"gripper_z: {pathPoint.gripper_z:.3f}")
    print(f"gripper_R: {pathPoint.gripper_R:.3f}")
    print(f"gripper_P: {pathPoint.gripper_P:.3f}")
    print(f"gripper_Y: {pathPoint.gripper_Y:.3f}")
    print(f"gripper_rel_x: {pathPoint.gripper_rel_x:.3f}")
    print(f"gripper_rel_y: {pathPoint.gripper_rel_y:.3f}")
    print(f"gripper_rel_z: {pathPoint.gripper_rel_z:.3f}")
    print(f"gripper_rel_R: {pathPoint.gripper_rel_R:.3f}")
    print(f"gripper_rel_P: {pathPoint.gripper_rel_P:.3f}")
    print(f"gripper_rel_Y: {pathPoint.gripper_rel_Y:.3f}")
    print(f"planned_gripper_y: {pathPoint.planned_gripper_y:.3f}")
    print(f"planned_gripper_x: {pathPoint.planned_gripper_x:.3f}")
    print(f"planned_gripper_z: {pathPoint.planned_gripper_z:.3f}")
    print(f"planned_gripper_R: {pathPoint.planned_gripper_R:.3f}")
    print(f"planned_gripper_P: {pathPoint.planned_gripper_P:.3f}")
    print(f"planned_gripper_Y: {pathPoint.planned_gripper_Y:.3f}")
    print(f"ik_fail: {pathPoint.ik_fail:.3f}")


def rpy_to_quiver_uvw(roll, pitch, yaw):
    U = np.cos(yaw) * np.cos(pitch)
    V = np.sin(yaw) * np.cos(pitch)
    W = np.sin(pitch)
    return U, V, W


def traced(func, ignoredirs=None):
    """
    Decorates func such that its execution is traced, but filters out any
    Python code outside of the system prefix.
    https://drake.mit.edu/python_bindings.html#debugging-with-the-python-bindings
    """
    import functools
    import sys
    import trace
    if ignoredirs is None:
        ignoredirs = ["/usr", sys.prefix]
    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return tracer.runfunc(func, *args, **kwargs)

    return wrapped


def calc_dist_to_sol(p):
    """NOTE: requires that planned_gripper_... is the plan that gripper_... tried to achieve"""
    return np.sqrt((p["gripper_x"] - p["planned_gripper_x"]) ** 2 +
                   (p["gripper_y"] - p["planned_gripper_y"]) ** 2 +
                   (p["gripper_z"] - p["planned_gripper_z"]) ** 2)
