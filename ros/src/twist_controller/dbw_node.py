#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Lane
from geometry_msgs.msg import PoseStamped

import math
import numpy as np
from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # TODO: Create `Controller` object [done]
        self.controller = Controller(vehicle_mass=vehicle_mass,
                                    fuel_capacity=fuel_capacity,
                                    brake_deadband=brake_deadband,
                                    decel_limit=decel_limit,
                                    accel_limit=accel_limit,
                                    wheel_radius=wheel_radius,
                                    wheel_base=wheel_base,
                                    steer_ratio=steer_ratio,
                                    max_lat_accel=max_lat_accel,
                                    max_steer_angle=max_steer_angle)

        # TODO: Subscribe to all the topics you need to [done]
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb,queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb,queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb,queue_size=1)
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb,queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.cur_pose_cb,queue_size=1)

        self.current_vel = None
        self.curr_ang_vel = None
        self.dbw_enabled = None
        self.linear_vel = None
        self.angular_vel = None
        self.final_waypoints = None
        self.cur_pose = None
        self.throttle = self.brake = self.steer = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            if not None in (self.current_vel, self.linear_vel, self.angular_vel):
                if self.final_waypoints and len(self.final_waypoints)>3:
                    cte = self.get_cte(self.final_waypoints, self.cur_pose)
                    self.throttle, self.brake, self.steer = self.controller.control(self.current_vel,
                                                                        self.dbw_enabled,
                                                                        self.linear_vel,
                                                                        self.angular_vel,
                                                                        cte)
                else:
                    self.throttle = 0
                    self.brake = 700
                    self.steer = 0

            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steer)
            rate.sleep()

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def twist_cb(self, msg):
        self.linear_vel = 0.95*msg.twist.linear.x # to avoid exceed speed limit
        self.angular_vel = 0.80*msg.twist.angular.z

    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    # add final waypoints and current_pos cb
    def final_waypoints_cb(self, msg):
        # self.linear_vel = msg.waypoints[0].twist.twist.linear.x 
        self.final_waypoints = msg.waypoints

    def cur_pose_cb(self,msg):
        self.cur_pose=msg


    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def get_cte(self, waypoints, cur_pos):
        if (waypoints is not None) and (cur_pos is not None):
            origin = waypoints[0].pose.pose.position
            #convert to x,y list
            waypoints_xy_list=list(map(lambda waypoint: [waypoint.pose.pose.position.x,
                                                        waypoint.pose.pose.position.y],
                                            waypoints))
            n = len(waypoints)
            #vehicle coordinates
            shift_xy_list = waypoints_xy_list - np.array([origin.x, origin.y])

            #calculte the angle to rotate
            angle=np.arctan2(shift_xy_list[min(11,n-1),1],shift_xy_list[min(11,n-1),0])
            rotation_matrix=np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
            #coordinate transformation (rotation)
            rotation_xy_list= np.dot(shift_xy_list,rotation_matrix)

            #polynomial fit, default set degree to be 2. 
            degree = 2
            poly_coef = np.polyfit(rotation_xy_list[:,0],rotation_xy_list[:,1],degree)

            #calculate vehicle pose in transformed coordinate
            car_xy_list = np.array([cur_pos.pose.position.x, cur_pos.pose.position.y]) 
            shift_car_pos = car_xy_list - np.array([origin.x, origin.y])
            rotation_car_pos = np.dot(shift_car_pos,rotation_matrix)
                
            #expected y value
            y_est = np.polyval(poly_coef, rotation_car_pos[0])
            y_actual = rotation_car_pos[1]

            cte = y_est - y_actual
        else:
            cte=0

        return cte


if __name__ == '__main__':
    DBWNode()
