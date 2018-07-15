
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# parameter tunning- throttle PID
Kp_v = 0.3#0.3
Ki_v = 0.1#0.1
Kd_v = 0.01#0.0
# parameter tunning- steering PID
Kp_s = 0.15#0.15
Ki_s = 0.001
Kd_s = 0.1#0.1

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
    wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement 
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        min_throttle = 0.0
        max_throttle = 0.7
        #self.throttle_controller = PID(kp=Kp_v, ki=Ki_v, kd=Kd_v,
        #                                mn=min_throttle,
        #                                mx=max_throttle)
        self.throttle_controller = PID(kp=Kp_v, ki=Ki_v, kd=Kd_v,
                                        mn=decel_limit, 
                                        mx=accel_limit)
        self.steering_controller = PID(kp=Kp_s, ki=Ki_s, kd=Kd_s,
                                        mn=-1.0*max_steer_angle,
                                        mx=1.0*max_steer_angle) 

        self.vel_lpf = LowPassFilter(tau=0.6, ts=0.02)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel, cte):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if (not dbw_enabled) or (abs(current_vel<1e-5)) or (abs(linear_vel)<1e-5):
            self.throttle_controller.reset()
            self.steering_controller.reset()
            return 0.0, 0.0, 0.0

        #current_vel = self.vel_lpf.filt(current_vel)

        #throttle
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        
        #brake
        brake = 0 
        if linear_vel==0. and current_vel<0.1:
            throttle = 0
            brake = 700
        elif throttle<0.1 and vel_error <0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius if abs(decel) > self.brake_deadband else 0.

        #steering
        predict_steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        steering_err = self.steering_controller.step(cte, sample_time)
        steering = predict_steering + steering_err

        #return throttle, brake, predict_steering
        return throttle, brake, steering_err
