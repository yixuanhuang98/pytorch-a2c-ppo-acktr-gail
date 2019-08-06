import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs,make_env
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.MBmodelProb import Model
from evaluation import evaluate
from a2c_ppo_acktr.model import Net



def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                         args.gamma, args.log_dir, device, False)
    # envs = make_env(args.env_name, args.seed,
    #                      args.gamma, args.log_dir,  False)
    print('lod_dir')
    print(args.log_dir)
    # actor_critic_safe, ob_rms = \
    #         torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    net = Net()
    #model_dict=torch.load("./pred_model/nn2.pt")
    pred_model = torch.load(os.path.join("./trained_models/mb", args.env_name + ".pt"))
            
    

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            net,
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    max_value = -1

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # print('state')
            # print(obs.size())
            # print(rollouts.obs)
            # print('action')
            # print(action.size())
            input_layer = torch.cat([obs,action], 1)
            #print(input_layer.size())
            
            next_state_pred = pred_model.predict(input_layer, deterministic=True)

            # Obser reward and next obs
            # print('action')
            # print(action)
            # print('obs')
            # print(obs.size())
            if(abs(obs[0][1].item()) > 1):
            # mpc function
                # print('enter mpc')
                # print(obs[0][1].item())
                if(abs(obs[0][1].item()) > max_value):
                    max_value = abs(obs[0][1].item())
                if(max_value == abs(obs[0][1].item())):
                    print(max_value)
                test_cases = 1000
                action_horizon = 10
                action_size = 2
                total_reward_list = []
                total_reward = 0
                test_action = np.random.uniform(-1,1,(test_cases,action_horizon,1,action_size))
                test_action.astype(float)
                # for i in range(test_cases):
                #     for j in range(action_horizon):
                #         action = torch.from_numpy(test_action[i,j])
                #         action = action.float()
                #         # print(action.dtype)
                #         # print(obs.dtype)
                #         input_layer_state = torch.cat([obs,action], 1)
                #         obs = pred_model.predict(input_layer_state, deterministic=True)
                #         #obs_data = obs.numpy()
                #         # we might consider some rewards at now
                #         total_reward += 10*(-5-obs[0][0].item())
                #         total_reward += -10000*(abs(obs[0][1].item()))
                #     total_reward_list.append(total_reward)
                #index = total_reward_list.index(max(total_reward_list))
                # action_numpy = test_action[index, 0]
                # action = torch.from_numpy(action_numpy)
            # reward = 10*(-5-carpos[0]) # the reward at the racecar and might do something with field
            
            # end mpc function
            
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            print('next_obs')
            print(obs)
            print('next_state_pred')
            print(next_state_pred)
            rollouts.insert(obs, next_state_pred, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, prediction_loss = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

            # torch.save([
            #     actor_critic,
            #     getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            # ], '/home/guest/pytorch-a2c-ppo-acktr-gail/trained_models/ppo/RacecarBulletEnv-v0-init-1-2.pt')

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            print('prediction loss')
            print(prediction_loss)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
