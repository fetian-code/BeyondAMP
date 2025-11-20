from isaaclab.utils import configclass

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from ...amp_env_cfg import AMPEnvCfg

from beyondAMP.amp_obs_grp import AMPObsBaiscCfg, AMPObsSoftTrackCfg, AMPObsHardTrackCfg


@configclass
class G1FlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
@configclass
class G1FlatEnvBasicCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsBaiscCfg()

@configclass
class G1FlatEnvSoftTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsSoftTrackCfg()
        
@configclass
class G1FlatEnvHardTrackCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AMPObsHardTrackCfg()