import numpy as np
import torch
import yaml
import os
import open3d as o3d
import trimesh













def process_robot_arm_bbox(bbox_list,bboxes,arm_model):
    """
    Process the robot arm bounding box.

    Args:
        bbox_list: the bounding box list
        bboxes: the bounding box
        arm_model: the arm model

    Returns:
        updated bounding box list
    """
    if arm_model == "default":
        for i in range(len(bbox_list)):
            if i==0:
                    bboxes[9]=bbox_list[i]
            elif i==1:
                    bboxes[1]=bbox_list[i]
            elif i==2:
                    bboxes[2]=bbox_list[i]
            elif i==3:
                    bboxes[3]=bbox_list[i]
            elif i==4:
                    bboxes[4]=bbox_list[i]
            elif i==5:  
                    bboxes[5]=bbox_list[i]
            elif i==6:
                    bboxes[6]=bbox_list[i]
            elif i==7:
                    bboxes[7]=bbox_list[i]
        return bboxes
    else:
        return bboxes
    


def process_robot_gripper_bbox(bbox_list,bboxes,arm_model,gripper_model):
    """
    Process the robot gripper bounding box.

    Args:
        bbox_list: the bounding box list
        bboxes: the bounding box
        gripper_model: the gripper model

    Returns:
        updated bounding box list
    """
    if gripper_model == "default" and arm_model == "default":
        for i in range(len(bbox_list)):
            if i==0:
                bboxes[9]=bbox_list[i]
            elif i==1:
                bboxes[1]=bbox_list[i]
            elif i==2:
                bboxes[2]=bbox_list[i]
            elif i==3:
                bboxes[3]=bbox_list[i]
            elif i==4:
                bboxes[4]=bbox_list[i]
            elif i==5:  
                bboxes[5]=bbox_list[i]
            elif i==6:
                bboxes[6]=bbox_list[i]
            elif i==7:
                bboxes[7]=bbox_list[i]
            elif i==8:
                bboxes[10]=bbox_list[i]
            elif i==9:
                bboxes[11]=bbox_list[i]
            elif i==10:
                bboxes[12]=bbox_list[i]
            elif i==11:
                bboxes[13]=bbox_list[i]

        return bboxes
    else:
        return bboxes
    

def process_robot_gripper_object_bbox(bbox_list,bboxes,arm_model,gripper_model):
    """
    Process the robot gripper object bounding box.

    Args:
        bbox_list: the bounding box list
        bboxes: the bounding box
        gripper_model: the gripper model

    Returns:
        updated bounding box list
    """
    if gripper_model == "default" and arm_model == "default":
        for i in range(len(bbox_list)):
                    if i==0:
                        bboxes[9]=bbox_list[i]
                    elif i==1:
                        bboxes[1]=bbox_list[i]
                    elif i==2:
                        bboxes[2]=bbox_list[i]
                    elif i==3:
                        bboxes[3]=bbox_list[i]
                    elif i==4:
                        bboxes[4]=bbox_list[i]
                    elif i==5:  
                        bboxes[5]=bbox_list[i]
                    elif i==6:
                        bboxes[6]=bbox_list[i]
                    elif i==7:
                        bboxes[7]=bbox_list[i]
                    elif i==8:
                        bboxes[8]=bbox_list[i]
                    elif i==9:
                        bboxes[10]=bbox_list[i]
                    elif i==10:
                        bboxes[11]=bbox_list[i]
                    elif i==11:
                        bboxes[12]=bbox_list[i]
                    elif i==12:
                        bboxes[13]=bbox_list[i]  
        return bboxes
    else:
        return bboxes
    

def expand_bbox(bboxes):
    """
    Expand the bounding box by hand if needed 
    """
    # for manual bbox fix for the novelpose part when the manual bbox is not accurate


    # use a knn to make all point in region that belongs to background to its nerest point with sam id
    bboxes[3,1] = bboxes[3,1]+0.015   # expand the bounding box by 10% to ensure all points are included
    bboxes[3,3] = bboxes[3,3]+0.025 
    bboxes[3,0] = bboxes[3,0]*1.05
    bboxes[3,2] = bboxes[3,2]*1.05
    bboxes[3,4] = bboxes[3,4]*1.05
    bboxes[3,5] = bboxes[3,5]*1.05  
    bboxes[2,0] = bboxes[2,0]
    bboxes[2,3] = bboxes[2,3]+0.03
    bboxes[2,2] = bboxes[2,2]
    bboxes[2,5] = bboxes[2,5]+0.03

    bboxes[4,4] = bboxes[4,4]+0.015
    bboxes[4,3] = bboxes[4,3]+0.02
            
    bboxes[5,0] = bboxes[5,0]
    bboxes[5,3] = bboxes[5,3]+0.02
    bboxes[5,4] = bboxes[5,4]+0.035
    bboxes[:4,2] = bboxes[:4,2]*1.05

    bboxes[4:6,1]  = bboxes[4:6,1]*1.05

    return bboxes