from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp


@configclass
class AMPObsBaiscCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    
AMPObsBaiscTerms = ["joint_pos", "joint_vel"]
    
@configclass
class AMPObsSoftTrackCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.base_lin_vel)
    body_ang_vel_w = ObsTerm(func=mdp.base_ang_vel)

AMPObsSoftTrackTerms = ["joint_pos", "joint_vel", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]

@configclass
class AMPObsHardTrackCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_pos_w = ObsTerm(func=mdp.body_pose_w)
    body_quat_w = ObsTerm(func=mdp.body_quat_w)
    body_lin_vel_w = ObsTerm(func=mdp.base_lin_vel)
    body_ang_vel_w = ObsTerm(func=mdp.base_ang_vel)
    
AMPObsHardTrackTerms = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]