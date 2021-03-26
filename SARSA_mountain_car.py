import gym
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


def greedy(Q_state):
    return np.argmax(Q_state)

def eps_greedy(Q_state,eps,nActions):
    if np.random.random() >= eps:
        action = np.argmax(Q_state)
    else:
        action = np.random.randint(0,nActions)
    return action

def get_discrete_state(state, observation_space_low, discrete_bins_size):
    discrete_state = (state - observation_space_low)/discrete_bins_size
    return tuple(discrete_state.astype(int))

def eps_decay(eps_0, episode, epsDecayEpisode):
    eps = eps_0 * np.power(0.2,(episode/epsDecayEpisode))
    return eps

def main():

    #####PARAMETERS##############
    EPISODES = 20000
    BINS = 20
    SEED = 42
    EPS_0 = 0.9
    ALPHA = 0.1 ##lr
    GAMMA = 0.95 ##Discount Rate
    EPS_DECAY_EPISODE = EPISODES * 0.25
    STATS_FREQ = 100
    RENDER_FREQ = 10000
    #############################

    env = gym.make('MountainCar-v0')
    env.seed(SEED)
    np.random.seed(SEED)

    nActions = env.action_space.n

    ## Initalize the Q-values matrix and structures to log the rewards
    Q = np.random.uniform(low=-2, high=0, size=[BINS,BINS,nActions])
    rewardsList = []
    ep_rewards_aggr = {'ep':[],'avg':[],'max':[],'min':[]}
    ## Initalize eps
    eps = EPS_0
    eps_list = []
    ## Discrete bins sizes
    discret_bins_shape = [BINS, BINS]
    discrete_bins_size = (env.observation_space.high - env.observation_space.low)/discret_bins_shape

    ##Start to experience
    for episode in range(EPISODES):
        observation = env.reset()
        ## Get starting state discretized
        state = get_discrete_state(observation,env.observation_space.low,discrete_bins_size)
        action = eps_greedy(Q[state],eps,nActions)
        ## Initialize Reward per episode
        totReward = 0
        done = False

        while not done:
            ## Do first experience
            observation, reward, done, _ = env.step(action)
            ## Keep track of Reward
            totReward += reward
            ## Get state reached after first action
            state2 = get_discrete_state(observation,env.observation_space.low,discrete_bins_size)
            ## Get alternative optimal action following greedy policy
            action2 = eps_greedy(Q[state2],eps,nActions)
            ## Update Q table with the alternative best action 
            Q[state + (action,)] = (1-ALPHA) * Q[state + (action,)] + ALPHA * ( reward + GAMMA * Q[state2 + (action2,)]) 
            ## Place your car in the new state
            state = state2
            action = action2

        rewardsList.append(totReward)

        if not episode % STATS_FREQ:
            avg_reward = sum(rewardsList[-STATS_FREQ:])/STATS_FREQ
            ep_rewards_aggr['ep'].append(episode)
            ep_rewards_aggr['avg'].append(avg_reward)
            ep_rewards_aggr['max'].append(max(rewardsList[-STATS_FREQ:]))
            ep_rewards_aggr['min'].append(min(rewardsList[-STATS_FREQ:]))
        
        ## Decay Epsilon
        eps_list.append(eps)
        eps = eps_decay(EPS_0, episode, EPS_DECAY_EPISODE)

    ep_rewards_aggr['avg'][0]=-200
    env.close()

    ## Plot results
    fig,ax = plt.subplots(figsize=(8,5))
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.plot(ep_rewards_aggr['ep'], ep_rewards_aggr['avg'], label="average reward")
    ax.plot(ep_rewards_aggr['ep'], ep_rewards_aggr['max'], label="max reward")
    ax.plot(ep_rewards_aggr['ep'], ep_rewards_aggr['min'], label="min reward")
    ax.set_title(f"SARSA(0) Rewards for bin={BINS}, alpha={ALPHA} gamma={GAMMA}, eps0={EPS_0}, decay_episode={EPS_DECAY_EPISODE}")
    ax2 = ax.twinx()
    ax2.set_ylabel("Epsilon")
    ax2.plot(eps_list, label='Episilon Decay behaviour', alpha=0.35, color='r')
    plt.figlegend(loc=4)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"./Mountain_Car_SARSA_Rewards for bin={BINS}, alpha={ALPHA}, gamma={GAMMA}, eps0={EPS_0}, decay_episode={EPS_DECAY_EPISODE}.png")

    ## Print avg reward over last 100 episodes
    print(f"Average reward over last 100 episode : {np.mean(rewardsList[-100:])}")

if __name__ == "__main__":
    main()