#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool, String
from detection_interfaces.msg import DetectionMessage
from enum import Enum

class RaceState(Enum):
    LANE_FOLLOWING = 1
    SIGN_DETECTED = 2
    TURNING_LEFT = 3
    TURNING_RIGHT = 4

class RaceController(Node):
    def __init__(self):
        super().__init__('race_controller')

        # self.sign_cooldown_time = self.get_parameter('sign_cooldown_time').value
        
        self.sub_lane_error = self.create_subscription(
            Float64, '/lane_error', self.cbLaneError, 1
        )
        
        self.sub_detection = self.create_subscription(
            DetectionMessage, '/detections', self.cbDetection, 1
        )
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        # self.pub_state = self.create_publisher(String, '/race_state', 1)
        self.pub_lane_control_active = self.create_publisher(Bool, '/lane_control_active', 1)

        self.current_state = RaceState.LANE_FOLLOWING
        self.detected_sign = None
        self.sign_confidence_threshold = 0.85        
        
        self.lane_error = 0.0
        
        self.start_turning_time = None
        self.turning_step = 0
    
        self.timer_control = self.create_timer(0.05, self.control_loop) 

    def cbLaneError(self, msg):
        self.lane_error = msg.data
        
    def cbDetection(self, msg):
        if msg.confidence >= self.sign_confidence_threshold:
            return
        
        if self.current_state == RaceState.LANE_FOLLOWING:
            self.detected_sign = msg.class_name
            self.current_state = RaceState.SIGN_DETECTED
            self.get_logger().info(f"Sign detected: {self.detected_sign}")

    def control_loop(self):
        # twist = Twist()

        if self.current_state == RaceState.LANE_FOLLOWING:
            self.pub_lane_control_active.publisth(Bool(data=True))

        elif self.current_state == RaceState.SIGN_DETECTED:
            self.start_turning_time = self.get_clock().now()

            if self.detected_sign == 'left':
                self.current_state = RaceState.TURNING_LEFT
            if self.detected_sign == "right":
                self.current_state = RaceState.TURNING_RIGHT

            # twist.linear.x = 0.2
            # twist.angular.z = 0.0

            self.pub_lane_control_active.publisth(Bool(data=False))
            # self.pub_cmd_vel.publish(twist)
            
        elif self.current_state == RaceState.TURNING_LEFT:  
            self.pub_lane_control_active.publisth(Bool(data=False))
            twist = self.execute_turn(self.start_turning_time)
            self.pub_cmd_vel.publish(twist)

        elif self.current_state == RaceState.TURNING_RIGHT:  
            self.pub_lane_control_active.publisth(Bool(data=False))
            twist = self.execute_turn(self.start_turning_time)
            self.pub_cmd_vel.publish(twist)

    def execute_turn(self, start_time):
        twist = Twist()

        if self.turning_step == 0:
            twist.linear.x = 0.2
            twist.angular.z = 0.0

            cur_time = self.get_clock().now()
            time_elapsed = (cur_time - start_time).nanoseconds / 1e9
            if time_elapsed > 1.5: 
                self.turning_step = 1
                self.get_logger().info('Entering intersection')

        elif self.turning_step == 1:
            twist.linear.x = 0.2
            if self.current_state == RaceState.TURNING_LEFT:
                twist.angular.z = 0.8
            elif self.current_state == RaceState.TURNING_RIGHT:
                twist.angular.z = -0.8

            cur_time = self.get_clock().now()
            time_elapsed = (cur_time - start_time).nanoseconds / 1e9
            if time_elapsed > 0.5:
                self.turning_step = 2
                self.get_logger().info('Turning')

        elif self.turning_step == 2:
            twist.linear.x = 0.3
            twist.angular.z = 0.0

            cur_time = self.get_clock().now()
            time_elapsed = (cur_time - start_time).nanoseconds / 1e9
            if time_elapsed > 1.0:
                self.turning_step = 3
                self.get_logger().info('Straigthening')

        elif self.turning_step == 3:
            # self.current_state = RaceState.LANE_FOLLOWING
            # self.turning_step = 0
            # self.get_logger().info('Completed turn')
            Kp = 0.8
            twist.linear.x = 0.4
            twist.angular.z = -Kp * self.lane_error
            
            if abs(self.lane_error) < 0.15:
                self.current_state = RaceState.LANE_FOLLOWING
                self.turning_step = 0
                self.get_logger().info('Completed turn')

        return twist
    
def main(args=None):
    rclpy.init(args=args)
    node = RaceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()