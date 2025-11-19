from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def amp_obs_body_displacement(env:ManagerBasedRLEnv, asset_cfg:SceneEntityCfg=SceneEntityCfg("robot")):
    return