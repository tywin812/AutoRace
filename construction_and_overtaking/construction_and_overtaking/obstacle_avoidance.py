#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
import numpy as np


class ObstacleAvoidance(Node):
    # State machine states
    STATE_NORMAL = 0
    STATE_OBSTACLE_DETECTED = 1
    STATE_AVOIDING_LEFT = 2
    STATE_AVOIDING_RIGHT = 3
    STATE_RETURNING = 4

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Subscriptions
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        self.sub_left_distance = self.create_subscription(
            Float64,
            '/lane_left_distance',
            self.left_distance_callback,
            10
        )
        self.sub_right_distance = self.create_subscription(
            Float64,
            '/lane_right_distance',
            self.right_distance_callback,
            10
        )
        self.sub_lane_state = self.create_subscription(
            UInt8,
            '/lane_detection_state',
            self.lane_state_callback,
            10
        )

        # Publications
        self.pub_avoid_cmd = self.create_publisher(
            Twist,
            '/avoid_control',
            10
        )
        self.pub_avoid_active = self.create_publisher(
            Bool,
            '/avoid_active',
            10
        )

        # State variables
        self.state = self.STATE_NORMAL
        self.left_distance = 999.0  # Distance to yellow line
        self.right_distance = 999.0  # Distance to white line
        self.lane_state = 0  # 0=none, 1=left only, 2=both, 3=right only
        self.front_distance = 999.0
        self.left_clear = True
        self.right_clear = True

        # Parameters
        self.obstacle_threshold = 1.0  # meters
        self.clear_threshold = 1.5  # meters to consider obstacle passed
        self.lane_margin = 100.0  # pixels - minimum space to lane before avoiding
        self.avoidance_speed = 0.15  # m/s
        self.avoidance_angular = 0.3  # rad/s

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Obstacle Avoidance Node initialized')

    def lidar_callback(self, scan_msg):
        """Process LiDAR data to detect obstacles."""
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')  # Replace 0 with inf

        # Front sector: ±30 degrees (assuming 360-degree scan)
        # Adjust indices based on your LiDAR configuration
        front_start = len(ranges) * 330 // 360
        front_end = len(ranges) * 30 // 360
        
        if front_end < front_start:
            front_ranges = np.concatenate([ranges[front_start:], ranges[:front_end]])
        else:
            front_ranges = ranges[front_start:front_end]

        self.front_distance = np.min(front_ranges) if len(front_ranges) > 0 else 999.0

        # Left sector: 60-120 degrees
        left_start = len(ranges) * 60 // 360
        left_end = len(ranges) * 120 // 360
        left_ranges = ranges[left_start:left_end]
        self.left_clear = np.min(left_ranges) > 0.5 if len(left_ranges) > 0 else True

        # Right sector: 240-300 degrees
        right_start = len(ranges) * 240 // 360
        right_end = len(ranges) * 300 // 360
        right_ranges = ranges[right_start:right_end]
        self.right_clear = np.min(right_ranges) > 0.5 if len(right_ranges) > 0 else True

    def left_distance_callback(self, msg):
        """Receive distance to left (yellow) lane."""
        self.left_distance = msg.data

    def right_distance_callback(self, msg):
        """Receive distance to right (white) lane."""
        self.right_distance = msg.data

    def lane_state_callback(self, msg):
        """Receive lane detection state."""
        self.lane_state = msg.data

    def control_loop(self):
        """Main control loop for state machine."""
        # State machine logic
        if self.state == self.STATE_NORMAL:
            self.handle_normal_state()
        elif self.state == self.STATE_OBSTACLE_DETECTED:
            self.handle_obstacle_detected_state()
        elif self.state == self.STATE_AVOIDING_LEFT:
            self.handle_avoiding_left_state()
        elif self.state == self.STATE_AVOIDING_RIGHT:
            self.handle_avoiding_right_state()
        elif self.state == self.STATE_RETURNING:
            self.handle_returning_state()

    def handle_normal_state(self):
        """Normal lane following mode."""
        # Publish that avoidance is inactive
        avoid_active = Bool()
        avoid_active.data = False
        self.pub_avoid_active.publish(avoid_active)

        # Check for obstacles
        if self.front_distance < self.obstacle_threshold:
            self.state = self.STATE_OBSTACLE_DETECTED
            self.get_logger().info('Obstacle detected! Deciding avoidance direction...')

    def handle_obstacle_detected_state(self):
        """Decide which direction to avoid."""
        # Decision logic based on lane distances
        if self.left_distance < self.lane_margin:
            # Not enough space on left, go right
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info('Avoiding RIGHT (left lane too close)')
        elif self.right_distance < self.lane_margin:
            # Not enough space on right, go left
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info('Avoiding LEFT (right lane too close)')
        elif self.lane_state == 1:
            # Only left lane visible, go right
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info('Avoiding RIGHT (only left lane visible)')
        elif self.lane_state == 3:
            # Only right lane visible, go left
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info('Avoiding LEFT (only right lane visible)')
        else:
            # Both lanes visible and enough space - default left for overtaking
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info('Avoiding LEFT (default for overtaking)')

    def handle_avoiding_left_state(self):
        """Execute left avoidance maneuver."""
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = self.avoidance_angular  # Turn left
        self.pub_avoid_cmd.publish(twist)

        # Check if obstacle is passed
        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING
            self.get_logger().info('Obstacle passed, returning to lane...')

    def handle_avoiding_right_state(self):
        """Execute right avoidance maneuver."""
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = -self.avoidance_angular  # Turn right
        self.pub_avoid_cmd.publish(twist)

        # Check if obstacle is passed
        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING
            self.get_logger().info('Obstacle passed, returning to lane...')

    def handle_returning_state(self):
        """Return to center of lane."""
        # Check if both lanes are visible and distances are balanced
        if self.lane_state == 2:  # Both lanes visible
            left_right_diff = abs(self.left_distance - self.right_distance)
            if left_right_diff < 50.0:  # Within 50 pixels of center
                self.state = self.STATE_NORMAL
                self.get_logger().info('Returned to lane following')
                return

        # Continue returning
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed * 0.8
        
        # Steer towards center
        if self.left_distance < self.right_distance:
            twist.angular.z = -0.2  # Steer right
        else:
            twist.angular.z = 0.2  # Steer left
            
        self.pub_avoid_cmd.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
