#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool
from enum import Enum

from detection_interfaces.msg import DetectionMessage
from traffic_light_interfaces.msg import TrafficLightState


class RaceState(Enum):
    WAIT_FOR_GREEN = 0
    LANE_FOLLOWING = 1
    SIGN_DETECTED = 2
    APPROACHING_TURN = 3
    TURNING_LEFT = 4
    TURNING_RIGHT = 5
    APPROACHING_CONSTRUCTION = 6


class RaceController(Node):
    def __init__(self):
        super().__init__('race_controller')
        
        self.sub_detection = self.create_subscription(
            DetectionMessage, '/detections', self.cbDetection, 1
        )
        self.sub_traffic_light = self.create_subscription(
            TrafficLightState, '/traffic_light_state', self.cbTrafficLight, 1
        )
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_lane_control_active = self.create_publisher(Bool, '/lane_control_active', 1)
        self.pub_construction = self.create_publisher(Bool, '/construction_area', 1)
        self.current_state = RaceState.WAIT_FOR_GREEN
        self.detected_sign = None
        
        self.sign_confidence_threshold = 0.85
        self.sign_area_threshold = 0.04  
        
        self.step_start_time = None
        self.turning_step = 0

        self.timer_control = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info(
            f'Race Controller initialized. Area threshold: {self.sign_area_threshold}'
        )

    def cbDetection(self, msg):
        
        if msg.confidence < self.sign_confidence_threshold:
            return
        
        if self.current_state not in [RaceState.SIGN_DETECTED, RaceState.APPROACHING_CONSTRUCTION]:
            return
        
        sign_area = msg.width * msg.height
        in_trigger_zone = sign_area > self.sign_area_threshold
        
        self.get_logger().info(
            f"Sign '{msg.class_name}': "
            f"conf={msg.confidence:.2f}, "
            f"area={sign_area:.3f}, "
            f"state={self.current_state.name}, "
            f"trigger={'YES' if in_trigger_zone else 'NO'}"
        )
        
        if self.current_state == RaceState.LANE_FOLLOWING:
            self.detected_sign = msg.class_name
            self.current_state = RaceState.SIGN_DETECTED
            self.get_logger().info(f"Sign '{self.detected_sign}' detected!")
        
        elif self.current_state == RaceState.SIGN_DETECTED:
            if not in_trigger_zone:
                return
            elif msg.class_name in ['right', 'left']:
                self.current_state = RaceState.APPROACHING_TURN
            elif msg.class_name == 'construction':
                self.current_state = RaceState.APPROACHING_CONSTRUCTION

    def cbTrafficLight(self, msg):
        if self.current_state != RaceState.WAIT_FOR_GREEN:
            return

        if msg.state == TrafficLightState.GREEN:
            self.get_logger().info("GREEN light received - starting race!")
            self.current_state = RaceState.LANE_FOLLOWING

    def control_loop(self):
        if self.current_state == RaceState.WAIT_FOR_GREEN:
            self.pub_lane_control_active.publish(Bool(data=False))
            return
        
        elif self.current_state == RaceState.LANE_FOLLOWING:
            self.pub_lane_control_active.publish(Bool(data=True))
        
        elif self.current_state == RaceState.SIGN_DETECTED:
            self.pub_lane_control_active.publish(Bool(data=True))
        
        elif self.current_state == RaceState.APPROACHING_TURN:
            self.pub_lane_control_active.publish(Bool(data=False))
            
            if self.detected_sign == 'left':
                self.step_start_time = self.get_clock().now()
                self.current_state = RaceState.TURNING_LEFT
                self.turning_step = 0
                self.get_logger().info("Starting LEFT turn")
            
            elif self.detected_sign == 'right':
                self.step_start_time = self.get_clock().now()
                self.current_state = RaceState.TURNING_RIGHT
                self.turning_step = 0
                self.get_logger().info("Starting RIGHT turn")
            
            twist = Twist()
            twist.linear.x = 0.2
            twist.angular.z = 0.0
            self.pub_cmd_vel.publish(twist)
        
        elif self.current_state == RaceState.TURNING_LEFT:
            twist = self.execute_turn()
            self.pub_cmd_vel.publish(twist)
        
        elif self.current_state == RaceState.TURNING_RIGHT:
            twist = self.execute_turn()
            self.pub_cmd_vel.publish(twist)

        elif self.current_state == RaceState.APPROACHING_CONSTRUCTION:
            self.pub_construction.publish(Bool(data=True))

    def execute_turn(self):
        twist = Twist()
        current_time = self.get_clock().now()
        elapsed = (current_time - self.step_start_time).nanoseconds / 1e9
        
        turn_dir = 1.0 if self.current_state == RaceState.TURNING_LEFT else -1.0
        
        if self.turning_step == 0:
            twist.linear.x = 0.1
            twist.angular.z = 0.0
            
            if elapsed > 0.8:
                self.turning_step = 1
                self.step_start_time = current_time
                self.get_logger().info('Entering intersection')
        
        elif self.turning_step == 1:
            twist.linear.x = 0.02
            twist.angular.z = 0.9 * turn_dir
            
            if elapsed > 1.0:
                self.turning_step = 2
                self.step_start_time = current_time
                self.get_logger().info('Turning')
        
        elif self.turning_step == 2:
            self.current_state = RaceState.LANE_FOLLOWING
            self.turning_step = 0
            self.step_start_time = None
            self.pub_lane_control_active.publish(Bool(data=True))
            self.get_logger().info('Turn completed!')
        
        return twist
    
    def shut_down(self):
        self.get_logger().info('Shutting down. Stopping robot.')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)

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
