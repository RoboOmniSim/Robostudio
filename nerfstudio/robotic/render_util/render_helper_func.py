from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch



def filter_with_semantic(semantic_id,assigned_ids,mark_id,xyz,opacities,scales,features_extra,rots,features_dc,index=0):
    semantic_id_ind=(semantic_id==assigned_ids[index]).reshape(-1)
    semantic_id_ind_sam=semantic_id[semantic_id==assigned_ids[index]].reshape(-1,1)
        
    select_xyz= np.array(xyz[semantic_id_ind])
    select_opacities=  np.array(opacities[semantic_id_ind])
    select_scales =  np.array(scales[semantic_id_ind])
    select_features_extra =  np.array(features_extra[semantic_id_ind])
    select_rotation =  np.array(rots[semantic_id_ind])
    select_feature_dc =  np.array(features_dc[semantic_id_ind])


    return select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam


