
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
    wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement 
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        min_throttle = 0.0
        max_throttle = 2.0
        self.throttle_controller = PID(kp=5, ki=0.02, kd=0.2,
                                        mn=min_throttle,
                                        mx=max_throttle)

        self.steering_controller = PID(2, 0.03, 1, mn=-max_steer_angle, mx=max_steer_angle)

        self.vel_lpf = LowPassFilter(tau=0.2, ts=0.1)
        #self.t_lpf = LowPassFilter(tau = 3, ts = 1)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.steering_controller.reset()
            return 0.0, 0.0, 0.0

        #current_vel = self.vel_lpf.filt(current_vel)

        #steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        #steering = self.vel_lpf.filt(steering)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        current_time = rospy.get_time()
        duration = current_time - self.last_time
        sample_time = duration + 1e-6 # to avoid division by zero
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        throttle = self.vel_lpf.filt(throttle)
        brake = 0 

        if linear_vel==0. and current_vel<0.1:
            throttle = 0
            brake = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius * abs(self.decel_limit)
        elif throttle<0.1 and vel_error <0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius
            
        err_steer = self.steering_controller.step(angular_vel, sample_time)
        steering_predict = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        steering = err_steer + steering_predict

        return throttle, brake, steering
