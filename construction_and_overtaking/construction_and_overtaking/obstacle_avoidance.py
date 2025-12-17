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
        self.obstacle_width = 0.0  # Width of detected obstacle
        self.left_clear = True
        self.right_clear = True
        self.obstacle_type = "NONE"  # NONE, CONE, CAR

        # Parameters
        self.obstacle_threshold = 1.0  # meters
        self.clear_threshold = 1.5  # meters to consider obstacle passed
        self.lane_margin = 100.0  # pixels - minimum space to lane before avoiding
        self.avoidance_speed = 0.15  # m/s
        self.avoidance_angular = 0.3  # rad/s
        self.cone_width_threshold = 0.15  # meters - objects narrower than this are cones
        self.car_width_threshold = 0.4  # meters - objects wider than this are cars

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Timer for status logging
        self.status_timer = self.create_timer(1.0, self.log_status)

        self.get_logger().info('Obstacle Avoidance Node initialized')
        self.get_logger().info('Waiting for obstacles...')

    def lidar_callback(self, scan_msg):
        """Process LiDAR data to detect obstacles."""
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')  # Replace 0 with inf

        # Front sector: ±30 degrees (assuming 360-degree scan)
        front_start = len(ranges) * 330 // 360
        front_end = len(ranges) * 30 // 360
        
        if front_end < front_start:
            front_ranges = np.concatenate([ranges[front_start:], ranges[:front_end]])
        else:
            front_ranges = ranges[front_start:front_end]

        self.front_distance = np.min(front_ranges) if len(front_ranges) > 0 else 999.0

        # Estimate obstacle width by counting consecutive close points
        if self.front_distance < self.obstacle_threshold:
            close_points = front_ranges < self.obstacle_threshold
            if np.any(close_points):
                # Find clusters of close points
                clusters = self.find_clusters(close_points)
                if len(clusters) > 0:
                    # Get the largest cluster
                    largest_cluster = max(clusters, key=len)
                    # Estimate width: number of points * angular resolution * distance
                    angular_res = scan_msg.angle_increment
                    self.obstacle_width = len(largest_cluster) * angular_res * self.front_distance
                    
                    # Determine obstacle type
                    if self.obstacle_width < self.cone_width_threshold:
                        self.obstacle_type = "CONE"
                    elif self.obstacle_width > self.car_width_threshold:
                        self.obstacle_type = "CAR"
                    else:
                        self.obstacle_type = "OBJECT"
        else:
            self.obstacle_type = "NONE"
            self.obstacle_width = 0.0

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

    def find_clusters(self, binary_array):
        """Find consecutive True values in binary array."""
        clusters = []
        current_cluster = []
        
        for i, val in enumerate(binary_array):
            if val:
                current_cluster.append(i)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters

    def left_distance_callback(self, msg):
        """Receive distance to left (yellow) lane."""
        self.left_distance = msg.data

    def right_distance_callback(self, msg):
        """Receive distance to right (white) lane."""
        self.right_distance = msg.data

    def lane_state_callback(self, msg):
        """Receive lane detection state."""
        self.lane_state = msg.data

    def log_status(self):
        """Periodically log current status."""
        if self.obstacle_type != "NONE":
            self.get_logger().info(
                f'[DETECTION] Obstacle: {self.obstacle_type} | '
                f'Distance: {self.front_distance:.2f}m | '
                f'Width: {self.obstacle_width:.2f}m'
            )
        else:
            self.get_logger().info(
                f'[CLEAR] No obstacles detected | '
                f'Front distance: {self.front_distance:.2f}m'
            )

    def control_loop(self):
        """Main control loop for state machine."""
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
        avoid_active = Bool()
        avoid_active.data = False
        self.pub_avoid_active.publish(avoid_active)

        if self.front_distance < self.obstacle_threshold:
            self.state = self.STATE_OBSTACLE_DETECTED
            self.get_logger().warn(
                f'[ALERT] {self.obstacle_type} detected at {self.front_distance:.2f}m! '
                f'Width: {self.obstacle_width:.2f}m'
            )

    def handle_obstacle_detected_state(self):
        """Decide which direction to avoid."""
        if self.left_distance < self.lane_margin:
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info(
                f'[DECISION] Avoiding {self.obstacle_type} to the RIGHT '
                f'(left lane too close: {self.left_distance:.0f}px)'
            )
        elif self.right_distance < self.lane_margin:
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info(
                f'[DECISION] Avoiding {self.obstacle_type} to the LEFT '
                f'(right lane too close: {self.right_distance:.0f}px)'
            )
        elif self.lane_state == 1:
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info(f'[DECISION] Avoiding {self.obstacle_type} to the RIGHT (only left lane visible)')
        elif self.lane_state == 3:
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info(f'[DECISION] Avoiding {self.obstacle_type} to the LEFT (only right lane visible)')
        else:
            self.state = self.STATE_AVOIDING_LEFT
            if self.obstacle_type == "CAR":
                self.get_logger().info(f'[DECISION] Overtaking {self.obstacle_type} on the LEFT')
            else:
                self.get_logger().info(f'[DECISION] Avoiding {self.obstacle_type} to the LEFT')

    def handle_avoiding_left_state(self):
        """Execute left avoidance maneuver."""
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = self.avoidance_angular
        self.pub_avoid_cmd.publish(twist)

        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING
            self.get_logger().info(f'[SUCCESS] {self.obstacle_type} passed! Returning to lane...')

    def handle_avoiding_right_state(self):
        """Execute right avoidance maneuver."""
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = -self.avoidance_angular
        self.pub_avoid_cmd.publish(twist)

        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING
            self.get_logger().info(f'[SUCCESS] {self.obstacle_type} passed! Returning to lane...')

    def handle_returning_state(self):
        """Return to center of lane."""
        if self.lane_state == 2:
            left_right_diff = abs(self.left_distance - self.right_distance)
            if left_right_diff < 50.0:
                self.state = self.STATE_NORMAL
                self.get_logger().info('[COMPLETE] Returned to normal lane following')
                return

        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed * 0.8
        
        if self.left_distance < self.right_distance:
            twist.angular.z = -0.2
        else:
            twist.angular.z = 0.2
            
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
