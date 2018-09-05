from custom_gym.unity import unity_env

class KobukiEnv(unity_env.UnityEnv):
    def __init__(self):
        super(KobukiEnv, self).__init__(app_name='RLkobuki.x86_64', idx=0)
    
