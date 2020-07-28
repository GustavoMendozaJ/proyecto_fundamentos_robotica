#!/usr/bin/env python
#

import rospy
from sensor_msgs.msg import JointState

from markers import *
from functions import *


rospy.init_node("testForwardKine")
pub = rospy.Publisher('joint_states', JointState, queue_size=10)
bmarker = BallMarker(color['GREEN'])

# Joint names
jnames = ['joint1', 'joint2', 'joint3',
          'joint4', 'joint5']
# Joint Configuration
q = np.array([0, 0, 0, 0, 0])

# End effector with respect to the base
T = robot_fkine(q)
print( np.round(T, 3) )
bmarker.position(T)

# Object (message) whose type is JointState
jstate = JointState()
# Set values to the message
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames
# Add the head joint value (with value 0) to the joints
jstate.position = q

# Loop rate (in Hz)
rate = rospy.Rate(100)
# Continuous execution loop
while not rospy.is_shutdown():
    # Current time (needed for ROS)
    jstate.header.stamp = rospy.Time.now()
    # Publish the message
    pub.publish(jstate)
    bmarker.publish()
    # Wait for the next iteration
    rate.sleep()
