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
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.MBmodelProb import Model
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

mu_array = np.zeros((8,1))
sigma_array = np.zeros((8,1))

def normalize(horizon_array,i):
    mu = np.mean(horizon_array)
    sigma = np.std(horizon_array)
    mu_array[i] = mu
    sigma_array[i] = sigma
    horizon_array = (horizon_array - mu) / sigma
    return horizon_array

def load(data_dir, args):
    # Ss = np.load(data_dir+"obs_"+args.env_name+".npy")
    # As = np.load(data_dir+"ac_"+args.env_name+".npy")
    #array = np.loadtxt('/Users/huangyixuan/txt_result/t')
    #array = np.loadtxt('/home/gao-4144/txt_result/test_1')
    #array = np.loadtxt('/Users/huangyixuan/txt_result/halfcheetah_test')
    array = np.loadtxt('/Users/huangyixuan/txt_result/racecar_7_new')
    # for i in range(8):
    #     array[:,i] = normalize(array[:,i],i)
    print(array.shape)
    Ss = array[:,:6]
    As = array[:,-2:]
    # print(Ss)
    # print(As)
    # print(Ss.shape())
    # print(As.shape())
    #Ds = np.load(data_dir+"done_"+args.env_name+".npy")

    Ss = np.array(np.reshape(Ss,(Ss.shape[0],-1)))
    As = np.array(np.reshape(As,(As.shape[0],-1)))
    # print(Ss.shape())
    # print(As.shape())
    #Ds = np.reshape(Ds,(Ds.shape[0],-1))

    #index = np.ravel(np.argwhere(Ds))


    S = Ss[:-1,:]
    A = As[:-1,:]
    #index = index + 1
    S_primes = Ss[1:,:]
    SA = np.concatenate((S,A),1)
    
    # print('s_prime shape')
    # print(S_primes.shape())
    # print('Sa shape')
    # print(np.shape(SA))

    return SA, S_primes

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    using_cuda = torch.cuda.is_available()

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    loss_criterian = nn.MSELoss()


    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, None, device, False)

    model = Model(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    model.to(device)

    start = time.time()

    SAs, SPs = load("./mbdata/",args)

    num_updates = 100
    batch_size = 512
    num_batch = SAs.shape[0] // batch_size

    length = SAs.shape[0]
    shuffle = np.random.permutation(length)
    SAs = SAs[shuffle]
    SPs = SPs[shuffle]

    train_index = num_batch // 10 * 9 * batch_size

    test_SAs = SAs[train_index:]
    test_SPs = SPs[train_index:]
    SAs= SAs[:train_index]
    SPs = SPs[:train_index]

    errors = []


    for i in range(num_updates):
        optimizer = optim.Adam(model.parameters(), lr=0.01/(i+1), eps=1e-5)
        length = SAs.shape[0]
        shuffle = np.random.permutation(length)
        SAs = SAs[shuffle]
        SPs = SPs[shuffle]
        losses = []

        for j in range(train_index // batch_size -1 ):

            if using_cuda:
                x = torch.tensor(SAs[j*batch_size:(j+1)*batch_size]).float().cuda()
                y = torch.tensor(SPs[j*batch_size:(j+1)*batch_size]).float().cuda()
            else:
                x = torch.tensor(SAs[j*batch_size:(j+1)*batch_size]).float()
                y = torch.tensor(SPs[j*batch_size:(j+1)*batch_size]).float()

            log_prob = model.evaluate(x,y)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if using_cuda:
            x = torch.tensor(test_SAs).float().cuda()
            y = torch.tensor(test_SPs).float().cuda()
        else:
            x = torch.tensor(test_SAs).float()
            y = torch.tensor(test_SPs).float()

        test_error = loss_criterian(model.predict(x, deterministic=True), y)

        errors.append(test_error)
        # if len(errors) > 20:
        #     if errors[-1] > errors[-2] and  errors[-2] > errors[-3]:
        #         break


        print("iteration:{}, loss:{}, test error: {}".format(i, np.asarray(losses).mean(), test_error))

    torch.save(
        model,
     os.path.join("./trained_models/mb", args.env_name + ".pt"))

if __name__ == "__main__":
    main()
