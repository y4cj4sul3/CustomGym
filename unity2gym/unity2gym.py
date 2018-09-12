from unity2gym.unity_env_v0 import UnityEnvV0
from unity2gym.unity_env_v1 import UnityEnvV1

# Unity Env with position information - v0
# ----------------------------------------
class UnityDiversityColor_v0(UnityEnvV0):
    def __init__(self, idx=0):
        idx = idx + 128
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_color/kobuki.x86_64", idx=idx)

class UnityDiversityColorObj_v0(UnityEnvV0):
    def __init__(self, idx=1):
        idx = idx + 128
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_colorObj/kobuki.x86_64", idx=idx)

class UnityDiversityColor2_v0(UnityEnvV0):
    def __init__(self, idx=2):
        idx = idx + 128
        super().__init__("/home/recharrs/UnityExec/UnityDiversity_color-2/kobuki.x86_64", idx=idx)

class UnityDiversitySkillRL_v0(UnityEnvV0):
    def __init__(self, idx=3):
        idx = idx + 128
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill/kobuki.x86_64", idx=idx)

# ----------------------------------------
class UnityDiversitySkillRL_v1(UnityEnvV1):
    def __init__(self, idx=13):
        idx = idx + 128
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill/kobuki.x86_64", idx=idx)

# ----------------------------------------
class UnityDiversitySkillRL_v2(UnityEnvV0):
    ''' coord, rgb skill. with reward shaping '''
    def __init__(self, idx=13):
        idx = idx + 128
        super().__init__("/home/recharrs/UnityExec/UnityRL_skill_v2/kobuki.x86_64", idx=idx)

