#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
import numpy as np
from numpy import *
from math import atan2, acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def T0_1(q):
    T = np.array([[cos(q), -sin(q), 0, 0],
                  [sin(q), cos(q), 0, 0],
                  [0, 0, 1, .75],
                  [0, 0, 0, 1 ]], dtype=np.float64)
    return T

def T1_2(q):
    T = np.array([[sin(q), cos(q), 0, .35],
                  [0, 0, 1, 0],
                  [cos(q), -sin(q), 0, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    return T

def T2_3(q):
    T = np.array([[cos(q), -sin(q), 0, 1.25],
                  [sin(q), cos(q), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    return T

def T3_4(q):
    T = np.array([[cos(q), -sin(q), 0, .054],
                  [0, 0, 1, 1.5],
                  [-sin(q), -cos(q), 0, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    return T

def T4_5(q):
    T = np.array([[cos(q), -sin(q), 0, 0],
                  [0, 0, -1, 0],
                  [sin(q), cos(q), 0, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    return T

def T5_6(q):
    T = np.array([[cos(q), -sin(q), 0, 0],
                  [0, 0, 1, 0],
                  [-sin(q), -sin(q), 0, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    return T

def T6_EE(q):
    T = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, .303],
                 [0, 0, 0, 1]], dtype=np.float64)
    return T

def Rot_x(q):
    T = np.array([[1,0,0],
                  [0, cos(q), -sin(q)],
                  [0, sin(q), cos(q)]], dtype=np.float64)
    return T

def Rot_y(q):
    T = np.array([[cos(q), 0, sin(q)],
                  [0, 1, 0],
                  [-sin(q), 0, cos(q)]], dtype=np.float64)
    return T

def Rot_z(q):
    T = np.array([[cos(q), -sin(q), 0],
                  [sin(q), cos(q), 0],
                  [0, 0, 1]], dtype=np.float64)
    return T

def Rot_extrinsic_XYZ(z,y,x):
    T = np.array([[cos(z)*cos(y), cos(z)*sin(y)*sin(x)-sin(z)*cos(x), cos(z)*sin(y)*cos(x) + sin(z)*sin(x)],
                  [sin(z)*cos(y), sin(z)*sin(y)*sin(x)+cos(z)*cos(x), sin(z)*sin(y)*cos(x) - cos(z)*sin(x)],
                  [-sin(y), cos(y)*sin(x), cos(y)*cos(x)]], dtype=np.float64)
    return T

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        # Initialize service response
        joint_trajectory_list = []
        PX = []
        PY = []
        PZ = []
        PXcheck = []
        PYcheck = []
        PZcheck = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

	    # Extract end-effector position and orientation from request
	    # px,py,pz = end-effector position
	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            EE = np.array([px,py,pz], dtype = np.float64)

            Rot_EE = Rot_extrinsic_XYZ(yaw,pitch,roll)
            #R_zR_y = np.dot(Rot_z(yaw), Rot_y(pitch))
            #Rot_EE = np.dot(R_zR_y, Rot_x(roll))

            ### Your IK code here
	    # Compensate for rotation discrepancy between DH parameters and Gazebo
            Rot_Error = np.dot(Rot_z(np.pi), Rot_y(-np.pi/2))
            Rot_EE = np.dot(Rot_EE, Rot_Error)
	    #
	    # Calculate joint angles using Geometric IK method
        # Calculate wrist center
            WC = EE - 0.303*Rot_EE[:,2]
	    # project wrist center coordinates to the base frame to obtain theta_1
            theta1 = atan2(WC[1],WC[0])
        # find sides of projected triangle connecting joint 2 and 3 and WC for theta2 and theta3
            side_a = 1.501
            side_b = sqrt(pow((sqrt(WC[0] ** 2 + WC[1] ** 2) - 0.35), 2) + pow((WC[2] - 0.75), 2))
            side_c = 1.25
        # solve for angles a and b with inverse cosine
            angle_a = acos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2 * side_b * side_c))
            angle_b = acos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2 * side_a * side_c))

            theta2 = np.pi/2 - angle_a - atan2(WC[2] - 0.75, sqrt(WC[0] ** 2 + WC[1] ** 2) - 0.35)
            theta3 = np.pi/2 - (angle_b + 0.036)
        # Find theta4, theta5, and theta6 by using the equations of the orientation part of T3_6
            R0_2 = np.dot(T0_1(theta1)[0:3, 0:3], T1_2(theta2)[0:3, 0:3])
            R0_3 = np.dot(R0_2, T2_3(theta3)[0:3, 0:3])
            R3_6 = np.dot(R0_3.T, Rot_EE)

            if EE[2]-WC[2] >= 0.:
                theta5 = atan2(sqrt(R3_6[0,2] ** 2 + R3_6[2,2] ** 2), R3_6[1,2])
            elif EE[2]-WC[2] < 0.:
                theta5 = atan2(-sqrt(R3_6[0,2] ** 2 + R3_6[2,2] ** 2), R3_6[1,2])

            if sin(theta5) >= 0.:
                theta4 = atan2(R3_6[2,2], -R3_6[0,2])
                theta6 = atan2(-R3_6[1,1], R3_6[1,0])
            elif sin(theta5) < 0.:
                theta4 = atan2(-R3_6[2,2], R3_6[0,2])
                theta6 = atan2(R3_6[1,1], -R3_6[1,0])
            ###

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
	        joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
	        joint_trajectory_list.append(joint_trajectory_point)
            print("Calculated %s of %s" % (x + 1, len(req.poses)))

            # Below makes an error graph. Should comment out for testing pick and place.
            T0_2 = np.dot(T0_1(theta1), T1_2(theta2))
            T0_3 = np.dot(T0_2, T2_3(theta3))
            T0_4 = np.dot(T0_3, T3_4(theta4))
            T0_5 = np.dot(T0_4, T4_5(theta5))
            T0_6 = np.dot(T0_5, T5_6(theta6))
            T0_EE = np.dot(T0_6, T6_EE(0.))
            PX.append(px)
            PY.append(py)
            PZ.append(pz)
            PXcheck.append(T0_EE[0,3])
            PYcheck.append(T0_EE[1,3])
            PZcheck.append(T0_EE[2,3])


        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(PX, PY, PZ, c='r', marker='o', label='EE desired position')
        ax.scatter(PXcheck, PYcheck, PZcheck, c='b',marker='^', label='IK calculated position')
        plt.legend(loc=2)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
