#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


class HardcodedConstructionPath(Node):
    def __init__(self):
        super().__init__('hardcoded_construction_path')

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_construction_zone = self.create_subscription(
            Bool, '/construction_area', self.construction_zone_callback, 1
        )

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_lane_active = self.create_publisher(Bool, '/lane_control_active', 10)

        self.is_active = False
        self.current_pose = None
        self.start_pose = None
        self.phase = 0  
        
        self.trajectory = [
            {'type': 'forward', 'distance': 0.5, 'speed': 0.20},
            
            {'type': 'turn', 'angle': -2.1, 'speed': 0.1, 'angular': -0.6},
            
            {'type': 'forward', 'distance': 0.3, 'speed': 0.20},
            
            {'type': 'turn', 'angle': -1.7, 'speed': 0.1, 'angular': 0.6},
            
            {'type': 'forward', 'distance': 0.3, 'speed': 0.20},

            {'type': 'turn', 'angle': 1.7, 'speed': 0.1, 'angular': 0.6},

        ]
        
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('=== Захардкоженная траектория через конусы ===')

    def construction_zone_callback(self, msg):
        if msg.data and not self.is_active:
            self.is_active = True
            self.phase = 0
            self.start_pose = None
            self.get_logger().info("🚧 Construction zone - Starting hardcoded path!")
        elif not msg.data and self.is_active:
            self.is_active = False
            self.get_logger().info("✓ Hardcoded path completed!")
            self.pub_cmd_vel.publish(Twist())
            self.pub_lane_active.publish(Bool(data=True))

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_distance_traveled(self):
        if not self.start_pose or not self.current_pose:
            return 0.0
        
        dx = self.current_pose.position.x - self.start_pose.position.x
        dy = self.current_pose.position.y - self.start_pose.position.y
        return math.sqrt(dx**2 + dy**2)

    def get_angle_rotated(self):
        if not self.start_pose or not self.current_pose:
            return 0.0
        
        start_yaw = self.quaternion_to_yaw(self.start_pose.orientation)
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
        
        angle_diff = current_yaw - start_yaw
        
        while angle_diff > math.pi:
            angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2.0 * math.pi
        
        return angle_diff

    def control_loop(self):
        if not self.is_active:
            return

        if self.current_pose is None:
            self.get_logger().warn('No odometry data!')
            return

        if self.phase >= len(self.trajectory):
            self.pub_lane_active.publish(Bool(data=True))
            self.get_logger().info('All phases completed - returning to lane following', 
                                   throttle_duration_sec=1.0)
            return

        current_phase = self.trajectory[self.phase]
        
        if self.start_pose is None:
            self.start_pose = self.current_pose
            self.get_logger().info(f'Phase {self.phase}: {current_phase["type"]} started')

        twist = Twist()

        if current_phase['type'] == 'forward':
            distance = self.get_distance_traveled()
            target_distance = current_phase['distance']
            
            if distance < target_distance:
                twist.linear.x = current_phase['speed']
                twist.angular.z = 0.0
                self.get_logger().info(
                    f'Phase {self.phase} FORWARD: {distance:.2f}m / {target_distance:.2f}m',
                    throttle_duration_sec=0.3
                )
            else:
                self.phase += 1
                self.start_pose = None
                self.get_logger().info(f'Phase {self.phase - 1} completed!')
                return

        elif current_phase['type'] == 'turn':
            angle = self.get_angle_rotated()
            target_angle = current_phase['angle']
            
            turn_dir = 1.0 if target_angle > 0 else -1.0
            
            if abs(angle) < abs(target_angle):
                twist.linear.x = current_phase['speed']
                twist.angular.z = turn_dir * current_phase['angular']
                self.get_logger().info(
                    f'Phase {self.phase} TURN: {math.degrees(angle):.1f}° / {math.degrees(target_angle):.1f}°',
                    throttle_duration_sec=0.3
                )
            else:
                self.phase += 1
                self.start_pose = None
                self.get_logger().info(f'Phase {self.phase - 1} completed!')
                return

        self.pub_cmd_vel.publish(twist)
        
        self.pub_lane_active.publish(Bool(data=False))


def main(args=None):
    rclpy.init(args=args)
    node = HardcodedConstructionPath()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
