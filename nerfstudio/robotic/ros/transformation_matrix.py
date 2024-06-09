#!/usr/bin/env python

import rospy
import tf2_ros
import numpy as np
import tf_conversions
import os
import rosbag


# this is for nerfstudio-02

def get_relative_tf(base, targets):
    tf_msgs = {}
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    while not rospy.is_shutdown():
        try:
            for target in targets:
                ts = buffer.lookup_transform(base, target, rospy.Time(0))
                tf_msgs[target] = ts
            return tf_msgs
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform between {} and {}".format(base, targets))
            # rospy.sleep(1.0)

def joint_status_callback(msg):
    # Print joint status
    print("Joint Names:", msg.name)
    print("Joint Positions:", msg.position)
    print("Joint Velocities:", msg.velocity)
    print("Joint Efforts:", msg.effort)
    print("")

def read_joint_status_from_rosbag(rosbag_file):
    # Open the ROS bag
    bag = rosbag.Bag(rosbag_file)

    # Iterate through messages in the bag
    for topic, msg, t in bag.read_messages(topics=['/joint_states']):
        # Call the joint status callback function
        joint_status_callback(msg)

    # Close the bag when finished
    # bag.close()





if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('joint_status_reader')

    # Provide the path to your ROS bag file
    rosbag_file = '/home/user/Downloads/2024-04-07-10-22-20.bag'

    # Read joint status from the ROS bag and print it
    read_joint_status_from_rosbag(rosbag_file)

def transform_to_matrix(transform):
    translation = [transform.transform.translation.x,
                   transform.transform.translation.y,
                   transform.transform.translation.z]
    rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]
    tf_matrix = tf_conversions.transformations.quaternion_matrix(rotation)
    tf_matrix[:3, 3] = translation
    return tf_matrix

def save_tf_data(tf_msgs):
    for target, tf_msg in tf_msgs.items():
        filename = target + "_tf.txt"
        with open(filename, 'a') as f:
            tf_matrix = transform_to_matrix(tf_msg)
            np.savetxt(f, tf_matrix, fmt='%.6f')

if __name__ == '__main__':
    rospy.init_node('tf_listener')
    base_frame = "base_link"  # Replace "base_frame" with the actual base frame ID

    # no gripper
    links = ['Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'Link6']

    while not rospy.is_shutdown():
        tf_msgs = get_relative_tf(base_frame, links)
        if tf_msgs:
            save_tf_data(tf_msgs)
    # first step：roscore
    # second step：rosbag play group1.bag 
    # run transformation_matrix：python transformation_matrix.py