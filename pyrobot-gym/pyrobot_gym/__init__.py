from gym.envs.registration import register

"""
for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }
"""
register(id='LocoBotReach-v1',
         entry_point='pyrobot_gym.tasks:LocoBotMujocoReachEnv',
         #kwargs=kwargs,
         max_episode_steps=50
)

register(id='LocoBotReach-v2',
         entry_point='pyrobot_gym.tasks:LocoBotGazeboReachEnv',
         #kwargs=kwargs,
         max_episode_steps=50
)

register(id='LocoBotReach-v3',
         entry_point='pyrobot_gym.tasks:LocoBotReachEnv',
         #kwargs=kwargs,
         max_episode_steps=50
)

register(id='LocoBotPush-v1',
         entry_point='pyrobot_gym.tasks:LocoBotMujocoPushEnv',
         #kwargs=kwargs,
         max_episode_steps=50
)

register(id='LocoBotPush-v2',
         entry_point='pyrobot_gym.tasks:LocoBotGazeboPushEnv',
         #kwargs=kwargs,
         max_episode_steps=50
)
