#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import Pose
import rosbag


def get_relative_pose(base, target):
    pose_msg = Pose()
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    while not rospy.is_shutdown():
        try:
            ts = buffer.lookup_transform(base, target, rospy.Time(0))
            pose_msg.position.x = ts.transform.translation.x
            pose_msg.position.y = ts.transform.translation.y
            pose_msg.position.z = ts.transform.translation.z
            pose_msg.orientation = ts.transform.rotation
            return pose_msg
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform between {} and {}".format(base, target))
            rospy.sleep(1.0)

# if __name__ == '__main__':
#     rospy.init_node('relative_pose_publisher')
#     base_frame = "base_link"  # Replace "base_frame" with the actual base frame ID
#     target_frame = "Link3"  # Replace "target_frame" with the actual target frame ID
#     relative_pose = get_relative_pose(base_frame, target_frame)
    
#     print("Relative Pose:", relative_pose)






