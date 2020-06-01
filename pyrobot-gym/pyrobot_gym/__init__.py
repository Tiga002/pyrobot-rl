from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

register(id='LocoBotReach-v1',
         entry_point='pyrobot_gym.tasks:LocoBotMujocoReachEnv',
         kwargs=kwargs,
         max_episode_steps=50
)
