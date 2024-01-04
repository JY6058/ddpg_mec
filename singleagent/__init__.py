from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='Multiagent-MEC-Envs-v0',
    entry_point='multiagent.envs:MECEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)
"""
register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
"""