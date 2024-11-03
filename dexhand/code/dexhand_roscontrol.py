#!/usr/bin/env python3
import rospy
from inspire_hand.srv import set_pos, set_angle, set_force, set_speed
import numpy as np

def call_service(service_name, matrix):
    try:
        # Services
        service_pos = rospy.ServiceProxy(service_name[0], set_pos)     # Range 0-2000
        service_angle = rospy.ServiceProxy(service_name[1], set_angle) # Range -1-1000
        service_force = rospy.ServiceProxy(service_name[2], set_force) # Range 0-1000
        service_speed = rospy.ServiceProxy(service_name[3], set_speed) # Range 0-1000

        # Pos
        req_pos = set_pos._request_class()
        req_pos.pos0 = int(matrix[0, 0])
        req_pos.pos1 = int(matrix[0, 1])
        req_pos.pos2 = int(matrix[0, 2])
        req_pos.pos3 = int(matrix[0, 3])
        req_pos.pos4 = int(matrix[0, 4])
        req_pos.pos5 = int(matrix[0, 5])
        
        response_1 = service_pos(req_pos)
        if response_1.pos_accepted:
            rospy.loginfo(f"Service {service_name} called successfully.")
        else:
            rospy.loginfo(f"Service {service_name} rejected the request.")

        # Angle
        req_angle = set_angle._request_class()
        req_angle.angle0 = int(matrix[1, 0])
        req_angle.angle1 = int(matrix[1, 1])
        req_angle.angle2 = int(matrix[1, 2])
        req_angle.angle3 = int(matrix[1, 3])
        req_angle.angle4 = int(matrix[1, 4])
        req_angle.angle5 = int(matrix[1, 5])

        response_2 = service_angle(req_angle)
        if response_2.angle_accepted:
            rospy.loginfo(f"Service {service_name} called successfully.")
        else:
            rospy.loginfo(f"Service {service_name} rejected the request.")
        
        # Force
        req_force = set_force._request_class()
        req_force.force0 = int(matrix[2, 0])
        req_force.force1 = int(matrix[2, 1])
        req_force.force2 = int(matrix[2, 2])
        req_force.force3 = int(matrix[2, 3])
        req_force.force4 = int(matrix[2, 4])
        req_force.force5 = int(matrix[2, 5])

        response_3 = service_force(req_force)
        if response_3.force_accepted:
            rospy.loginfo(f"Service {service_name} called successfully.")
        else:
            rospy.loginfo(f"Service {service_name} rejected the request.")

        # Speed
        req_speed = set_speed._request_class()
        req_speed.speed0 = int(matrix[3, 0])
        req_speed.speed1 = int(matrix[3, 1])
        req_speed.speed2 = int(matrix[3, 2])
        req_speed.speed3 = int(matrix[3, 3])
        req_speed.speed4 = int(matrix[3, 4])
        req_speed.speed5 = int(matrix[3, 5])

        response_4 = service_speed(req_speed)
        if response_4.speed_accepted:
            rospy.loginfo(f"Service {service_name} called successfully.")
        else:
            rospy.loginfo(f"Service {service_name} rejected the request.")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":

    matrices = np.array([
        [20, 20, 20, 20, 20, 20],
        [1, 1, 1, 1, 1, 1],
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10]
    ])

    services = [
        'inspire_hand/set_pos',
        'inspire_hand/set_angle',
        'inspire_hand/set_force',
        'inspire_hand/set_speed'
    ]

    call_service(services, matrices)
