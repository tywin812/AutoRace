#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool
from nav_msgs.msg import Odometry
from enum import Enum
import math

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
    CONSTRUCTION_ZONE = 7


class TurnPhase(Enum):
    MOVE_FORWARD = 0  
    ROTATE = 1  
    MOVE_OUT = 2        
    COMPLETE = 3      


class RaceController(Node):
    def __init__(self):
        super().__init__('race_controller')
        
        self.sub_detection = self.create_subscription(
            DetectionMessage, '/detections', self.cbDetection, 1
        )
        self.sub_traffic_light = self.create_subscription(
            TrafficLightState, '/traffic_light_state', self.cbTrafficLight, 1
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.cbOdometry, 1
        )
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_lane_control_active = self.create_publisher(Bool, '/lane_control_active', 1)
        self.pub_construction = self.create_publisher(Bool, '/construction_area', 1)
        
        self.current_state = RaceState.WAIT_FOR_GREEN
        self.detected_sign = None
        
        self.sign_confidence_threshold = 0.85
        self.sign_area_threshold = 0.04
        self.construction_sign_area_threshold = 0.05

        self.current_pose = None
        self.turn_start_pose = None
        
        self.turn_forward_distance = 0.18 
        self.turn_target_angle = math.pi / 2  
        self.turn_angle_tolerance = 0.1  
        
        self.forward_speed = 0.12   
        self.turn_linear_speed = 0.02 
        self.turn_angular_speed = 0.7

        self.exit_speed = 0.1
        self.turn_exit_distance = 0.05
        self.turn_phase = None
        
        self.turn_max_time = 10.0
        self.turn_timeout_start = None
        
        self.construction_start_pose = None
        self.construction_activation_distance = 0.8 
        
        self.last_construction_state = False
        
        self.timer_control = self.create_timer(0.05, self.control_loop)

    def cbOdometry(self, msg):
        self.current_pose = msg.pose.pose

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_distance_traveled(self):
        if not self.turn_start_pose or not self.current_pose:
            return 0.0
        
        dx = self.current_pose.position.x - self.turn_start_pose.position.x
        dy = self.current_pose.position.y - self.turn_start_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance
    
    def get_construction_distance(self):
        if not self.construction_start_pose or not self.current_pose:
            return 0.0
        
        dx = self.current_pose.position.x - self.construction_start_pose.position.x
        dy = self.current_pose.position.y - self.construction_start_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance

    def get_angle_rotated(self):
        if not self.turn_start_pose or not self.current_pose:
            return 0.0
        
        start_yaw = self.quaternion_to_yaw(self.turn_start_pose.orientation)
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
        
        angle_diff = current_yaw - start_yaw
        
        while angle_diff > math.pi:
            angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2.0 * math.pi
        
        return abs(angle_diff)

    def publish_construction_state(self, state):
        if state != self.last_construction_state:
            self.pub_construction.publish(Bool(data=state))
            self.last_construction_state = state
            self.get_logger().info(f'Construction area: {state}')

    def cbDetection(self, msg):
        if msg.confidence < self.sign_confidence_threshold:
            return
        
        sign_area = msg.width * msg.height
        
        if msg.class_name == 'construction':
            in_trigger_zone = sign_area > self.construction_sign_area_threshold
        else:
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
            self.publish_construction_state(False)
            return
        
        elif self.current_state == RaceState.LANE_FOLLOWING:
            self.pub_lane_control_active.publish(Bool(data=True))
            self.publish_construction_state(False)
        
        elif self.current_state == RaceState.SIGN_DETECTED:
            self.pub_lane_control_active.publish(Bool(data=True))
            self.publish_construction_state(False)
        
        elif self.current_state == RaceState.APPROACHING_TURN:
            self.pub_lane_control_active.publish(Bool(data=False))
            self.publish_construction_state(False)
            
            if self.detected_sign in ['left', 'right']:
                self.turn_phase = TurnPhase.MOVE_FORWARD
                self.turn_start_pose = self.current_pose
                self.turn_timeout_start = self.get_clock().now()
                
                if self.detected_sign == 'left':
                    self.current_state = RaceState.TURNING_LEFT
                    self.get_logger().info("Starting LEFT turn")
                else:
                    self.current_state = RaceState.TURNING_RIGHT
                    self.get_logger().info("Starting RIGHT turn")
        
        elif self.current_state in [RaceState.TURNING_LEFT, RaceState.TURNING_RIGHT]:
            self.publish_construction_state(False)
            twist = self.execute_turn()
            self.pub_cmd_vel.publish(twist)

        elif self.current_state == RaceState.APPROACHING_CONSTRUCTION:
            self.construction_start_pose = self.current_pose
            self.current_state = RaceState.CONSTRUCTION_ZONE
            self.publish_construction_state(False) 
            self.pub_lane_control_active.publish(Bool(data=True)) 
        
        elif self.current_state == RaceState.CONSTRUCTION_ZONE:
            distance = self.get_construction_distance()
            
            if distance < self.construction_activation_distance:
                self.pub_lane_control_active.publish(Bool(data=True))
                self.publish_construction_state(False)
                self.get_logger().debug(
                    f"Construction zone - lane following: {distance:.2f}m / {self.construction_activation_distance:.2f}m",
                    throttle_duration_sec=0.5
                )
            else:
                self.publish_construction_state(True)
                self.get_logger().debug(
                    f"Construction zone - obstacle avoidance ACTIVE (traveled {distance:.2f}m)",
                    throttle_duration_sec=1.0
                )

    def execute_turn(self):
        twist = Twist()
        
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available!')
            return twist
        
        # if self.turn_timeout_start:
        #     elapsed = (self.get_clock().now() - self.turn_timeout_start).nanoseconds / 1e9
        #     if elapsed > self.turn_max_time:
        #         self.get_logger().warn('Turn timeout! Forcing lane following')
        #         self.finish_turn()
        #         return Twist()
        
        turn_dir = 1.0 if self.current_state == RaceState.TURNING_LEFT else -1.0
        
        if self.turn_phase == TurnPhase.MOVE_FORWARD:
            distance = self.get_distance_traveled()
            
            twist.linear.x = self.forward_speed
            twist.angular.z = 0.0
            
            if distance >= self.turn_forward_distance:
                self.turn_phase = TurnPhase.ROTATE
                self.turn_start_pose = self.current_pose  
                self.get_logger().info(f'ROTATE - moved forward {distance:.3f}m')
        
        elif self.turn_phase == TurnPhase.ROTATE:
            angle = self.get_angle_rotated()
            
            twist.linear.x = self.turn_linear_speed
            twist.angular.z = self.turn_angular_speed * turn_dir
            
            if angle >= (self.turn_target_angle - self.turn_angle_tolerance):
                self.turn_phase = TurnPhase.MOVE_OUT
                self.turn_start_pose = self.current_pose  
                self.get_logger().info(f'MOVE_OUT - rotated {math.degrees(angle):.1f} degrees')

        elif self.turn_phase == TurnPhase.MOVE_OUT:
            distance = self.get_distance_traveled()
            
            twist.linear.x = self.exit_speed
            twist.angular.z = 0.0
            
            if distance >= self.turn_exit_distance:
                self.turn_phase = TurnPhase.COMPLETE
                self.get_logger().info(f'COMPLETE - moved out {distance:.3f}m')
    
        elif self.turn_phase == TurnPhase.COMPLETE:
            self.finish_turn()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        return twist

    def finish_turn(self):
        self.current_state = RaceState.LANE_FOLLOWING
        self.turn_phase = None
        self.turn_start_pose = None
        self.turn_timeout_start = None
        self.detected_sign = None
        self.pub_lane_control_active.publish(Bool(data=True))
        self.get_logger().debug('Turn completed - resuming lane following')
    
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
