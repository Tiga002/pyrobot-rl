import gym
import random
import gym
import os
import numpy as np

from environment_manager import RandomizedEnvironment
from agent import Agent
from replay_buffer import Episode, ReplayBuffer

EPISODES = 1000000

directory = "checkpoints"
experiment = "LocoBotPush-v1"
env = gym.make(experiment)

# Testing Hyperparameters
TESTING_INTERVAL = 200  # Number of updates between 2 evaluation of the policy
TESTING_ROLLOUTS = 50  # Number of rollouts performed to evaluate the current policy

# Algorithm Hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = 100
MAX_STEPS = 50
GAMMA = 0.99  # discount on future returns
K = 0.8  # probability of replay with H.E.R.

# Initialize the agent, both the actor/critic network
agent = Agent(experiment, BATCH_SIZE*MAX_STEPS)

# Initialize the environment sampler
randomized_environment = RandomizedEnvironment(experiment, [0.0, 1.0], [])

# Initialize the experience replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)

if not os.path.exists(directory):
    os.makedirs(directory)

for ep in range(EPISODES):
    print('================== Episode {} ==========================='.format(ep))
    # 1. Generate a environment with randomized parameters
    randomized_environment.sample_env()
    env, env_params = randomized_environment.get_env()

    # reset the new generated environment and get the initial obs
    current_obs_dict = env.reset()

    # extract the current goal from obs and initialize the episode
    goal = current_obs_dict['desired_goal']
    # episode instance, will then insert to the experience replay buffer
    episode = Episode(goal, env_params, MAX_STEPS)

    # Get the First observation and take a fake action(random action) at first
    obs = current_obs_dict['observation']
    achieved_goal = current_obs_dict['achieved_goal']
    last_action = env.action_space.sample()
    reward = env.compute_reward(achieved_goal, goal, 0)
    episode.add_step(last_action, obs, reward, achieved_goal)
    done = False

    # Rollout the whole episode
    #while not done:
    for t in range(MAX_STEPS):
        obs = current_obs_dict['observation']
        history = episode.get_history()
        noise = agent.action_noise()
        # behaviour policy = actor_policy + noise
        action_output = agent.evaluate_actor(agent._actor.predict, obs, goal, history) + noise
        # Step the Environment
        action = [action_output[0][0], action_output[0][1], action_output[0][2], action_output[0][3]]
        new_obs_dict, step_reward, done, info = env.step(action)

        new_obs = new_obs_dict['observation']
        achieved_goal = new_obs_dict['achieved_goal']
        # Add the transition tuple to the episode saver
        episode.add_step(action, new_obs, step_reward, achieved_goal, terminal=done)

        current_obs_dict = new_obs_dict

    # Store the episode into the experience replay buffer
    replay_buffer.add(episode)

    # Replay the experience replay buffer with H.E.R. with a probability k
    if random.random() < K:
        # replace the goal with the achieved goal
        new_goal = current_obs_dict['achieved_goal']
        # Create a new episode instance to hold the hindsight rollout
        replay_episode = Episode(new_goal, env_params, MAX_STEPS)
        for action, state, achieved_goal, done in zip(episode.get_actions(),
                                                      episode.get_states(),
                                                      episode.get_achieved_goals(),
                                                      episode.get_terminal()):
            # compute the new reward
            step_reward = env.compute_reward(achieved_goal, new_goal, 0)
            # Add the hindsighted fake transition
            replay_episode.add_step(action, state, step_reward, achieved_goal, terminal=done)

        # Add the hindsighted episode rollouts to the experience replay buffer
        replay_buffer.add(replay_episode)

    # Close the environment
    randomized_environment.close_env()

    # Perform a batch update of the network if we can sample a big enough batch of episodes from the buffer
    if replay_buffer.size() > BATCH_SIZE:
        # Sample a batch of episodes
        episodes = replay_buffer.sample_batch(BATCH_SIZE)

        state_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_state()])
        action_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_action()])
        next_state_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_state()])
        reward_batch = np.zeros([BATCH_SIZE*MAX_STEPS])
        env_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_env()])
        goal_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_goal()])
        history_batch = np.zeros([BATCH_SIZE*MAX_STEPS, MAX_STEPS, agent.get_dim_state()+agent.get_dim_action()])
        t_batch = []

        for i in range(BATCH_SIZE):
            # Put the states,actions,next_states,reward,env,goal,history from differnet episodes into the batch defined above
            state_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_states())[:-1]
            action_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_actions())[1:]
            next_state_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_states())[1:]
            reward_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_rewards())[1:]

            env_batch[i*MAX_STEPS:(i+1)*MAX_STEPS]=np.array(MAX_STEPS*[episodes[i].get_env()])
            goal_batch[i*MAX_STEPS:(i+1)*MAX_STEPS]=np.array(MAX_STEPS*[episodes[i].get_goal()])
            history_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array([episodes[i].get_history(t = t) for t in range(1, MAX_STEPS+1)])

            # WARNING FIXME: needs padding
            t_batch += episodes[i].get_terminal()[1:]

        target_action_batch = agent.evaluate_actor_batch(agent._actor.predict_target, next_state_batch, goal_batch, history_batch)
        predicted_actions = agent.evaluate_actor_batch(agent._actor.predict, next_state_batch, goal_batch, history_batch)
        target_q = agent.evaluate_critic_batch(agent._critic.predict_target, next_state_batch, predicted_actions, goal_batch, history_batch, env_batch)

        y_i = []
        for k in range(BATCH_SIZE*MAX_STEPS):
            if t_batch[k]:
                y_i.append(reward_batch[k])
            else:
                y_i.append(reward_batch[k] + GAMMA * target_q[k])

        predicted_q_value, _ = agent.train_critic(state_batch, action_batch, goal_batch, history_batch, env_batch, np.reshape(y_i, (BATCH_SIZE*MAX_STEPS, 1)))

        # Update the actor policy using the sampled gradient
        a_outs = agent.evaluate_actor_batch(agent._actor.predict, state_batch, goal_batch, history_batch)
        grads = agent.action_gradients_critic(state_batch, a_outs, goal_batch, history_batch, env_batch)
        agent.train_actor(state_batch, goal_batch, history_batch, grads[0])

        # Update target networks
        agent.update_target_actor()
        agent.update_target_critic()

    # perform policy evaluation
    if ep % TESTING_INTERVAL == 0 and ep != 0:
        success_number = 0
        print('Performing policy evalution ~~~~')

        for test_ep in range(TESTING_ROLLOUTS):
            randomized_environment.sample_env()
            env, env_params = randomized_environment.get_env()

            current_obs_dict = env.reset()

            # read the current goal, and initialize the episode
            goal = current_obs_dict['desired_goal']
            episode = Episode(goal, env_params, MAX_STEPS)

            # get the first observation and first fake "old-action"
            # TODO: decide if this fake action should be zero or random
            obs = current_obs_dict['observation']
            achieved = current_obs_dict['achieved_goal']
            last_action = env.action_space.sample()

            episode.add_step(last_action, obs, 0, achieved)

            done = False

            # rollout the whole episode
            #while not done:
            for t in range(MAX_STEPS):
                obs = current_obs_dict['observation']
                history = episode.get_history()

                action = agent.evaluate_actor(agent._actor.predict_target, obs, goal, history)

                new_obs_dict, step_reward, done, info = env.step(action[0])

                new_obs = new_obs_dict['observation']
                achieved = new_obs_dict['achieved_goal']

                episode.add_step(action[0], new_obs, step_reward, achieved, terminal=done)

                current_obs_dict = new_obs_dict

            if info['is_success'] > 0.0:
                success_number += 1

            randomized_environment.close_env()

        print("Testing at episode {}, success rate : {}".format(ep, success_number/TESTING_ROLLOUTS))
        agent.save_model("{}/ckpt_episode_{}".format(directory, ep))
        agent.update_success(success_number/TESTING_ROLLOUTS, ep)
        with open("csv_log.csv", "a") as csv_log:
            csv_log.write("{}; {}\n".format(ep, success_number/TESTING_ROLLOUTS))
