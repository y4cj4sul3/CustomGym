from gym.envs.registration import registry, register, make, spec

# Unity Env with position - v0
# ----------------------------------------
''' example: env = gym.make('UDColor-v0') '''

register(
    id='UDColor-v0',
    entry_point='unity2rllab.unity2rllab:UDColor_Position',
)
register(
    id='UDColorObj-v0',
    entry_point='unity2rllab.unity2rllab:UDColorObj_Position',
)
register(
    id='UDColor2-v0',
    entry_point='unity2rllab.unity2rllab:UDColor2_Position',
)

# Unity Env without position - v1
# ----------------------------------------

register(
    id='UDColor-v1',
    entry_point='unity2rllab.unity2rllab:UDColor_NoPosition',
)
register(
    id='UDColorObj-v1',
    entry_point='unity2rllab.unity2rllab:UDColorObj_NoPosition',
)
register(
    id='UDColor2-v1',
    entry_point='unity2rllab.unity2rllab:UDColor2_NoPosition',
)
