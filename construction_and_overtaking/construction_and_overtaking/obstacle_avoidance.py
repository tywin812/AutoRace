#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
import numpy as np


class ObstacleAvoidance(Node):
    STATE_NORMAL = 0
    STATE_OBSTACLE_DETECTED = 1
    STATE_AVOIDING_LEFT = 2
    STATE_AVOIDING_RIGHT = 3
    STATE_RETURNING = 4

    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        self.sub_lane_state = self.create_subscription(UInt8, '/lane_detection_state', self.lane_state_callback, 10)

        self.pub_avoid_cmd = self.create_publisher(Twist, '/avoid_control', 10)
        self.pub_avoid_active = self.create_publisher(Bool, '/avoid_active', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)

        self.state = self.STATE_NORMAL
        self.left_distance = 999.0
        self.right_distance = 999.0
        self.lane_state = 0
        self.front_distance = 999.0
        self.front_left_min = 999.0
        self.front_right_min = 999.0
        self.back_distance = 999.0
        self.left_min_dist = 999.0
        self.right_min_dist = 999.0
        self.obstacle_width = 0.0
        self.left_clear = True
        self.right_clear = True
        self.obstacle_type = "NONE"

        self.image_center_x = 320.0  # Assuming 640x480 resolution
        self.lane_width_pixels = 450.0  # Reduced to make calculated lane wider in meters
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.15  # Increased margin

        self.obstacle_threshold = 0.7
        self.clear_threshold = 1.5
        self.lane_margin = 100.0
        self.corridor_width = 0.5  # Increased from 0.3
        self.max_obstacle_width = 0.6
        self.avoidance_speed = 0.1
        self.avoidance_angular = 0.7
        self.cone_width_threshold = 0.15
        self.car_width_threshold = 0.4

        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.log_status)

        self.get_logger().info('Obstacle Avoidance Node initialized')
        self.get_logger().info(f'Lane filtering: {self.pixels_per_meter:.1f} px/m')

    def pixels_to_meters(self, pixel_distance):
        return pixel_distance / self.pixels_per_meter

    def get_lane_boundaries_in_meters(self):
        if self.left_distance < 999.0:
            left_boundary_m = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
        else:
            left_boundary_m = 999.0
        if self.right_distance < 999.0:
            right_boundary_m = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
        else:
            right_boundary_m = -999.0
        return left_boundary_m, right_boundary_m

    def is_point_within_lanes(self, y_position, left_boundary, right_boundary):
        if left_boundary >= 999.0 and right_boundary <= -999.0:
            return True
        if left_boundary < 999.0 and y_position > left_boundary:
            return False
        if right_boundary > -999.0 and y_position < right_boundary:
            return False
        return True

    def lidar_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')
        ranges[ranges == float('-inf')] = float('inf')

        ranges = np.roll(ranges, len(ranges)//2)
        angles = np.linspace(np.pi, 3*np.pi, len(ranges), endpoint=False)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        check_dist = self.obstacle_threshold + 0.5
        mask = (x > 0) & (x < check_dist) & (np.abs(y) < 1.0)
        
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        if self.lane_state > 0:
            within_lanes = np.array([self.is_point_within_lanes(y[i], left_boundary, right_boundary) for i in range(len(y))])
            mask = mask & within_lanes
        
        indices = np.where(mask)[0]
        
        self.front_distance = 999.0
        self.front_left_min = 999.0
        self.front_right_min = 999.0
        self.obstacle_type = "NONE"
        self.obstacle_width = 0.0
        
        if len(indices) > 0:
            clusters = []
            current_cluster = [indices[0]]
            for i in range(1, len(indices)):
                idx = indices[i]
                prev_idx = indices[i-1]
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
            
            min_dist = 999.0
            valid_obstacle_found = False
            
            for cluster in clusters:
                cluster_indices = np.array(cluster)
                cx = x[cluster_indices]
                cy = y[cluster_indices]
                in_corridor = (cx < self.obstacle_threshold) & (np.abs(cy) < self.corridor_width / 2)
                if np.any(in_corridor):
                    width = np.max(cy) - np.min(cy)
                    if width > self.max_obstacle_width:
                        continue
                    dist = np.min(cx)
                    if dist < min_dist:
                        min_dist = dist
                        self.obstacle_width = width
                        valid_obstacle_found = True
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

        ranges_orig = np.array(scan_msg.ranges)
        ranges_orig[ranges_orig == 0] = float('inf')
        
        left_start = len(ranges_orig) * 60 // 360
        left_end = len(ranges_orig) * 120 // 360
        left_ranges = ranges_orig[left_start:left_end]
        self.left_min_dist = np.min(left_ranges) if len(left_ranges) > 0 else 999.0
        self.left_clear = self.left_min_dist > 0.5

        right_start = len(ranges_orig) * 240 // 360
        right_end = len(ranges_orig) * 300 // 360
        right_ranges = ranges_orig[right_start:right_end]
        self.right_min_dist = np.min(right_ranges) if len(right_ranges) > 0 else 999.0
        self.right_clear = self.right_min_dist > 0.5

        back_start = len(ranges_orig) * 150 // 360
        back_end = len(ranges_orig) * 210 // 360
        back_ranges = ranges_orig[back_start:back_end]
        self.back_distance = np.min(back_ranges) if len(back_ranges) > 0 else 999.0

    def left_distance_callback(self, msg):
        self.left_distance = 999.0 if msg.data < 0 else msg.data

    def right_distance_callback(self, msg):
        self.right_distance = 999.0 if msg.data < 0 else msg.data

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def log_status(self):
        state_names = {0: "NORMAL", 1: "DETECTED", 2: "AVOID_LEFT", 3: "AVOID_RIGHT", 4: "RETURNING"}
        current_state_name = state_names.get(self.state, "UNKNOWN")
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        self.get_logger().info(f'State: {current_state_name} | Lanes: L={self.left_distance:.1f}px({left_boundary:.2f}m) R={self.right_distance:.1f}px({right_boundary:.2f}m) | LIDAR: F:{self.front_distance:.2f}m')

    def control_loop(self):
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
        avoid_active = Bool()
        avoid_active.data = False
        self.pub_avoid_active.publish(avoid_active)

        max_vel = Float64()
        slow_down_dist = 1.5
        if self.front_distance < slow_down_dist:
            scale = (self.front_distance - self.obstacle_threshold) / (slow_down_dist - self.obstacle_threshold)
            scale = max(0.0, min(1.0, scale))
            max_vel.data = 0.05 + (scale**0.5) * (0.22 - 0.05)
        else:
            max_vel.data = 0.22
        self.pub_max_vel.publish(max_vel)

        if self.front_distance < self.obstacle_threshold:
            self.state = self.STATE_OBSTACLE_DETECTED
            self.get_logger().warn(f'[ALERT] {self.obstacle_type} at {self.front_distance:.2f}m! Width: {self.obstacle_width:.2f}m (WITHIN LANES)')

    def handle_obstacle_detected_state(self):
        obstacle_on_left = self.front_left_min < self.front_right_min
        if self.left_distance < self.lane_margin:
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info(f'[DECISION] Forced RIGHT (Left: {self.left_distance})')
        elif self.right_distance < self.lane_margin:
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info(f'[DECISION] Forced LEFT (Right: {self.right_distance})')
        elif self.left_distance == 999.0 and self.right_distance < 600.0:
            self.state = self.STATE_AVOIDING_LEFT
            self.get_logger().info('[DECISION] Smart LEFT')
        elif self.right_distance == 999.0 and self.left_distance < 600.0:
            self.state = self.STATE_AVOIDING_RIGHT
            self.get_logger().info('[DECISION] Smart RIGHT')
        elif obstacle_on_left:
            self.state = self.STATE_AVOIDING_RIGHT
        else:
            self.state = self.STATE_AVOIDING_LEFT

    def handle_avoiding_left_state(self):
        if self.left_distance < self.lane_margin:
            self.get_logger().warn('[SAFETY] Abort LEFT! Switch RIGHT.')
            self.state = self.STATE_AVOIDING_RIGHT
            return
        self.get_logger().info('ACTION: LEFT', throttle_duration_sec=1.0)
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = self.avoidance_angular
        self.pub_avoid_cmd.publish(twist)
        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING

    def handle_avoiding_right_state(self):
        if self.right_distance < self.lane_margin:
            self.get_logger().warn('[SAFETY] Abort RIGHT! Switch LEFT.')
            self.state = self.STATE_AVOIDING_LEFT
            return
        self.get_logger().info('ACTION: RIGHT', throttle_duration_sec=1.0)
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        twist = Twist()
        twist.linear.x = self.avoidance_speed
        twist.angular.z = -self.avoidance_angular
        self.pub_avoid_cmd.publish(twist)
        if self.front_distance > self.clear_threshold:
            self.state = self.STATE_RETURNING

    def handle_returning_state(self):
        if self.front_distance < self.obstacle_threshold:
            self.state = self.STATE_OBSTACLE_DETECTED
            return
        if self.lane_state == 2:
            left_right_diff = abs(self.left_distance - self.right_distance)
            if left_right_diff < 50.0:
                self.state = self.STATE_NORMAL
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
