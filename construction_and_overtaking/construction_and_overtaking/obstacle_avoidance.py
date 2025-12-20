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
        self.pub_max_vel = self.create_publisher(
            Float64,
            '/control/max_vel',
            10
        )

        # State variables
        self.state = self.STATE_NORMAL
        self.left_distance = 999.0  # Distance to yellow line
        self.right_distance = 999.0  # Distance to white line
        self.lane_state = 0  # 0=none, 1=left only, 2=both, 3=right only
        self.front_distance = 999.0
        self.front_left_min = 999.0
        self.front_right_min = 999.0
        self.back_distance = 999.0
        self.left_min_dist = 999.0
        self.right_min_dist = 999.0
        self.obstacle_width = 0.0  # Width of detected obstacle
        self.left_clear = True
        self.right_clear = True
        self.obstacle_type = "NONE"  # NONE, CONE, CAR

        # Parameters
        self.obstacle_threshold = 0.5  # meters
        self.clear_threshold = 1.5  # meters to consider obstacle passed
        self.lane_margin = 100.0  # pixels - minimum space to lane before avoiding
        self.corridor_width = 0.3  # meters - width of the path to check for obstacles
        self.max_obstacle_width = 0.6  # meters - ignore objects wider than this (walls)
        self.avoidance_speed = 0.1  # m/s
        self.avoidance_angular = 0.7  # rad/s
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
        # Replace 0 and -inf with inf to ignore them
        ranges[ranges == 0] = float('inf')
        ranges[ranges == float('-inf')] = float('inf')

        # Roll ranges so Front is in the middle (index len/2)
        # This ensures continuous indices for the front sector
        # Original: 0=Front, 90=Left, 180=Back, 270=Right
        # Rolled: 180=Back, 270=Right, 360/0=Front, 90=Left
        ranges = np.roll(ranges, len(ranges)//2)
        angles = np.linspace(np.pi, 3*np.pi, len(ranges), endpoint=False)

        # Cartesian coordinates
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # 1. Filter points in a wider area to detect full object width
        # Look ahead slightly further than threshold to see the whole object
        check_dist = self.obstacle_threshold + 0.5
        # Look wider (e.g. +/- 1.0m) to see if it's a wall
        mask = (x > 0) & (x < check_dist) & (np.abs(y) < 1.0)
        indices = np.where(mask)[0]
        
        self.front_distance = 999.0
        self.front_left_min = 999.0
        self.front_right_min = 999.0
        self.obstacle_type = "NONE"
        self.obstacle_width = 0.0
        
        if len(indices) > 0:
            # Cluster points based on index continuity and distance
            clusters = []
            current_cluster = [indices[0]]
            for i in range(1, len(indices)):
                idx = indices[i]
                prev_idx = indices[i-1]
                
                # Check continuity
                idx_jump = idx - prev_idx
                p1 = np.array([x[prev_idx], y[prev_idx]])
                p2 = np.array([x[idx], y[idx]])
                dist = np.linalg.norm(p1 - p2)
                
                if idx_jump < 10 and dist < 0.3:
                    current_cluster.append(idx)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [idx]
            clusters.append(current_cluster)
            
            # Analyze clusters
            min_dist = 999.0
            valid_obstacle_found = False
            
            for cluster in clusters:
                cluster_indices = np.array(cluster)
                cx = x[cluster_indices]
                cy = y[cluster_indices]
                
                # Check if any part of cluster is in the critical corridor
                in_corridor = (cx < self.obstacle_threshold) & (np.abs(cy) < self.corridor_width / 2)
                
                if np.any(in_corridor):
                    # Calculate total width of the object (including parts outside corridor)
                    width = np.max(cy) - np.min(cy)
                    
                    if width > self.max_obstacle_width:
                        # Too wide -> Wall -> Ignore
                        continue
                        
                    # Valid obstacle
                    dist = np.min(cx)
                    if dist < min_dist:
                        min_dist = dist
                        self.obstacle_width = width
                        valid_obstacle_found = True
                        
                        # Determine side
                        min_x_idx = np.argmin(cx)
                        closest_y = cy[min_x_idx]
                        
                        if closest_y > 0:
                            self.front_left_min = dist
                            self.front_right_min = 999.0
                        else:
                            self.front_right_min = dist
                            self.front_left_min = 999.0
                            
            if valid_obstacle_found:
                self.front_distance = min_dist
                if self.obstacle_width < self.cone_width_threshold:
                    self.obstacle_type = "CONE"
                elif self.obstacle_width > self.car_width_threshold:
                    self.obstacle_type = "CAR"
                else:
                    self.obstacle_type = "OBJECT"

        # Left sector: 60-120 degrees
        # Note: We rolled the ranges, so we need to adjust indices for side sectors too
        # Or just use the original ranges for side sectors?
        # Easier to unroll or re-read ranges for side sectors to avoid confusion
        # But let's just use the rolled ranges with correct angles
        
        # Left is around 90 degrees (2.5pi in our rolled angles)
        # 2.5pi corresponds to index 3*len/4
        # 60-120 deg -> 2.33pi - 2.66pi
        
        left_start = int(len(ranges) * (2.33 - 1.0) / 2.0) # Map pi..3pi to 0..len
        # Actually, let's just use the original scan_msg for side sectors to be safe and simple
        ranges_orig = np.array(scan_msg.ranges)
        ranges_orig[ranges_orig == 0] = float('inf')
        
        left_start = len(ranges_orig) * 60 // 360
        left_end = len(ranges_orig) * 120 // 360
        left_ranges = ranges_orig[left_start:left_end]
        self.left_min_dist = np.min(left_ranges) if len(left_ranges) > 0 else 999.0
        self.left_clear = self.left_min_dist > 0.5

        # Right sector: 240-300 degrees
        right_start = len(ranges_orig) * 240 // 360
        right_end = len(ranges_orig) * 300 // 360
        right_ranges = ranges_orig[right_start:right_end]
        self.right_min_dist = np.min(right_ranges) if len(right_ranges) > 0 else 999.0
        self.right_clear = self.right_min_dist > 0.5

        # Back sector: 150-210 degrees
        back_start = len(ranges_orig) * 150 // 360
        back_end = len(ranges_orig) * 210 // 360
        back_ranges = ranges_orig[back_start:back_end]
        self.back_distance = np.min(back_ranges) if len(back_ranges) > 0 else 999.0

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
        if msg.data < 0:
            self.left_distance = 999.0
        else:
            self.left_distance = msg.data

    def right_distance_callback(self, msg):
        """Receive distance to right (white) lane."""
        if msg.data < 0:
            self.right_distance = 999.0
        else:
            self.right_distance = msg.data

    def lane_state_callback(self, msg):
        """Receive lane detection state."""
        self.lane_state = msg.data

    def log_status(self):
        """Periodically log current status."""
        # Only log if something interesting is happening or state changed
        # pass
        state_names = {
            0: "NORMAL",
            1: "DETECTED",
            2: "AVOID_LEFT",
            3: "AVOID_RIGHT",
            4: "RETURNING"
        }
        current_state_name = state_names.get(self.state, "UNKNOWN")
        
        self.get_logger().info(
            f'State: {current_state_name} | Lanes: L={self.left_distance:.1f} R={self.right_distance:.1f} | '
            f'LIDAR: F:{self.front_distance:.2f}m'
        )
        # if self.obstacle_type != "NONE":
        #     self.get_logger().info(
        #         f'[DETECTION] Obstacle: {self.obstacle_type} | '
        #         f'Distance: {self.front_distance:.2f}m | '
        #         f'Width: {self.obstacle_width:.2f}m'
        #     )
        # else:
        #     self.get_logger().info(
        #         f'[CLEAR] No obstacles detected | '
        #         f'Front distance: {self.front_distance:.2f}m'
        #     )

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

        # Dynamic speed control based on distance
        max_vel = Float64()
        # Start slowing down earlier (at 1.5m)
        slow_down_dist = 1.5
        if self.front_distance < slow_down_dist:
            # Scale speed based on distance
            # Min speed 0.05 at threshold, Max speed 0.22 at slow_down_dist
            # Use sqrt to slow down faster as we get closer
            scale = (self.front_distance - self.obstacle_threshold) / (slow_down_dist - self.obstacle_threshold)
            scale = max(0.0, min(1.0, scale))
            max_vel.data = 0.05 + (scale**0.5) * (0.22 - 0.05)
        else:
            max_vel.data = 0.22
            
        self.pub_max_vel.publish(max_vel)

        if self.front_distance < self.obstacle_threshold:
            self.state = self.STATE_OBSTACLE_DETECTED
            self.get_logger().warn(
                f'[ALERT] {self.obstacle_type} detected at {self.front_distance:.2f}m! '
                f'Width: {self.obstacle_width:.2f}m'
            )

    def handle_obstacle_detected_state(self):
        """Decide which direction to avoid."""
        # Check where the obstacle is relative to center
        obstacle_on_left = self.front_left_min < self.front_right_min
        
        # if obstacle_on_left:
        #      self.get_logger().info(f'[ANALYSIS] Obstacle is on the LEFT side (L:{self.front_left_min:.2f}m R:{self.front_right_min:.2f}m)')
        # else:
        #      self.get_logger().info(f'[ANALYSIS] Obstacle is on the RIGHT side (L:{self.front_left_min:.2f}m R:{self.front_right_min:.2f}m)')

        # Decision logic
        # Priority 1: Check lane margins (don't go off road)
        if self.left_distance < self.lane_margin:
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info(f'[DECISION] Forced RIGHT (Left lane too close: {self.left_distance})')
        elif self.right_distance < self.lane_margin:
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info(f'[DECISION] Forced LEFT (Right lane too close: {self.right_distance})')
        
        # Priority 2: Smart lane logic (if one lane is missing, go the other way)
        elif self.left_distance == 999.0 and self.right_distance < 600.0:
             self.state = self.STATE_AVOIDING_LEFT
             self.get_logger().info(f'[DECISION] Smart LEFT (Right lane visible {self.right_distance:.0f}, Left open)')
        elif self.right_distance == 999.0 and self.left_distance < 600.0:
             self.state = self.STATE_AVOIDING_RIGHT
             self.get_logger().info(f'[DECISION] Smart RIGHT (Left lane visible {self.left_distance:.0f}, Right open)')

        # Priority 3: Check obstacle position
        elif obstacle_on_left:
            # If obstacle is on the left, try to go right
            self.state = self.STATE_AVOIDING_RIGHT
            # self.get_logger().info(f'[DECISION] Avoiding {self.obstacle_type} to the RIGHT (obstacle on left)')
        else:
            # If obstacle is on the right, try to go left
            self.state = self.STATE_AVOIDING_LEFT
            # self.get_logger().info(f'[DECISION] Avoiding {self.obstacle_type} to the LEFT (obstacle on right)')

    def handle_avoiding_left_state(self):
        """Execute left avoidance maneuver."""
        # Safety check: if we are too close to the left lane, abort and go right
        if self.left_distance < self.lane_margin:
            self.get_logger().warn(f'[SAFETY] Aborting LEFT turn! Left lane too close ({self.left_distance}). Switching to RIGHT.')
            self.state = self.STATE_AVOIDING_RIGHT
            return

        self.get_logger().info('ACTION: Turning LEFT', throttle_duration_sec=1.0)
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = self.avoidance_angular
        self.pub_avoid_cmd.publish(twist)

        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING
            # self.get_logger().info(f'[SUCCESS] {self.obstacle_type} passed! Returning to lane...')

    def handle_avoiding_right_state(self):
        """Execute right avoidance maneuver."""
        # Safety check: if we are too close to the right lane, abort and go left
        if self.right_distance < self.lane_margin:
            self.get_logger().warn(f'[SAFETY] Aborting RIGHT turn! Right lane too close ({self.right_distance}). Switching to LEFT.')
            self.state = self.STATE_AVOIDING_LEFT
            return

        self.get_logger().info('ACTION: Turning RIGHT', throttle_duration_sec=1.0)
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)

        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = -self.avoidance_angular
        self.pub_avoid_cmd.publish(twist)

        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING
            # self.get_logger().info(f'[SUCCESS] {self.obstacle_type} passed! Returning to lane...')

    def handle_returning_state(self):
        """Return to center of lane."""
        # Check for new obstacles while returning
        if self.front_distance < self.obstacle_threshold:
            self.state = self.STATE_OBSTACLE_DETECTED
            # self.get_logger().warn(
            #     f'[ALERT] New {self.obstacle_type} detected while returning at {self.front_distance:.2f}m! '
            #     f'Switching to avoidance.'
            # )
            return

        if self.lane_state == 2:
            left_right_diff = abs(self.left_distance - self.right_distance)
            if left_right_diff < 50.0:
                self.state = self.STATE_NORMAL
                # self.get_logger().info('[COMPLETE] Returned to normal lane following')
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
