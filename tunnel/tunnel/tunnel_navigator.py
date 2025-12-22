#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool, Header, ColorRGBA
from geometry_msgs.msg import Twist, PoseStamped, Point, Vector3
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData, Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import time
import cv2


class TunnelNavigator(Node):
    """Навигация в тоннеле: поиск въезда и центрирование по стенам"""

    def __init__(self):
        super().__init__('tunnel_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        
        self.sub_left_path = self.create_subscription(Path, '/detect/lane_left_path', self.left_path_callback, 10)
        self.sub_right_path = self.create_subscription(Path, '/detect/lane_right_path', self.right_path_callback, 10)
        self.sub_pixel_counts = self.create_subscription(Point, '/detect/lane_pixel_counts', self.pixel_counts_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_tunnel_entered = self.create_publisher(Bool, '/tunnel/entered', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)
        
        self.pub_grid = self.create_publisher(OccupancyGrid, '/debug/grid', 10)
        self.pub_path = self.create_publisher(Path, '/debug/path', 10)
        self.pub_traversed_path = self.create_publisher(Path, '/tunnel/traversed_path', 10)
        self.pub_debug_markers = self.create_publisher(MarkerArray, '/tunnel/debug_markers', 10)

        # Состояние
        self.traversed_path = Path()
        self.left_distance = 999.0
        self.right_distance = 999.0
        self.white_pixels = 0.0
        self.yellow_pixels = 0.0
        self.current_path = []
        self.left_lane_path = []
        self.right_lane_path = []
        
        # Параметры сетки
        self.grid_resolution = 0.02
        self.grid_width = 2.0
        self.grid_length = 2.0
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Размеры робота
        self.robot_radius = 0.10
        self.safety_radius = 0.30
        
        self.lane_boundary_margin = 0.10
        
        # Цены сетки
        self.forbidden_cost = 1000.0
        self.obstacle_cost = 100.0
        
        # Управление
        self.speed = 0.08
        self.steering_gain = 1.0
        self.look_ahead_distance = 0.6
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.state = 'APPROACH'  # APPROACH, TUNNEL
        self.gate_center = None

        self.get_logger().info('=== Tunnel Navigator Started ===')

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        
        self.traversed_path.header = msg.header
        self.traversed_path.poses.append(pose)
        
        if len(self.traversed_path.poses) > 2000:
            self.traversed_path.poses.pop(0)
            
        self.pub_traversed_path.publish(self.traversed_path)

    def pixel_counts_callback(self, msg):
        self.white_pixels = msg.x
        self.yellow_pixels = msg.y

    def left_path_callback(self, msg):
        points = []
        for pose in msg.poses:
            points.append((pose.pose.position.x, pose.pose.position.y))
        self.left_lane_path = points

    def right_path_callback(self, msg):
        points = []
        for pose in msg.poses:
            points.append((pose.pose.position.x, pose.pose.position.y))
        self.right_lane_path = points

    def left_distance_callback(self, msg):
        self.left_distance = 999.0 if msg.data < 0 else msg.data

    def right_distance_callback(self, msg):
        self.right_distance = 999.0 if msg.data < 0 else msg.data

    def lidar_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        ranges[np.isnan(ranges)] = float('inf')
        ranges[ranges == 0] = float('inf')
        ranges[np.isneginf(ranges)] = float('inf')
        
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        
        if self.state == 'APPROACH':
            # Поиск въезда
            valid_ranges = ranges[ranges < 10.0]
            if len(valid_ranges) > 0:
                 self.get_logger().info(
                    f"Lidar: Min={np.min(valid_ranges):.2f}, Max={np.max(valid_ranges):.2f}. "
                    f"Angles: {np.min(angles):.2f} to {np.max(angles):.2f}",
                    throttle_duration_sec=2.0
                )
            
            # Левый сектор: +2.0 до +PI
            left_mask = (angles > 2.0) & (ranges < 3.0)
            # Правый сектор: -PI до -2.0
            right_mask = (angles < -2.0) & (ranges < 3.0)
            
            marker_array = MarkerArray()
            
            def create_sector_marker(id, min_angle, max_angle, r, g, b):
                marker = Marker()
                marker.header = scan_msg.header
                marker.ns = "gate_sectors"
                marker.id = id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale = Vector3(x=0.05, y=0.0, z=0.0)
                marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                
                p0 = Point(x=0.0, y=0.0, z=0.0)
                vis_range = 2.0
                
                p1 = Point()
                p1.x = vis_range * np.cos(min_angle)
                p1.y = vis_range * np.sin(min_angle)
                
                p2 = Point()
                p2.x = vis_range * np.cos(max_angle)
                p2.y = vis_range * np.sin(max_angle)
                
                marker.points.append(p0)
                marker.points.append(p1)
                marker.points.append(p2)
                marker.points.append(p0)
                
                return marker

            marker_array.markers.append(create_sector_marker(0, 2.0, 3.14, 0.0, 1.0, 0.0))
            marker_array.markers.append(create_sector_marker(1, -3.14, -2.0, 1.0, 0.0, 0.0))
            
            def create_points_marker(mask, id, r, g, b):
                marker = Marker()
                marker.header = scan_msg.header
                marker.ns = "gate_search"
                marker.id = id
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
                marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                
                if mask is None:
                    valid_r = ranges[ranges < 10.0]
                    valid_a = angles[ranges < 10.0]
                else:
                    valid_r = ranges[mask]
                    valid_a = angles[mask]
                
                for r_val, a_val in zip(valid_r, valid_a):
                    p = Point()
                    p.x = r_val * np.cos(a_val)
                    p.y = r_val * np.sin(a_val)
                    p.z = 0.0
                    marker.points.append(p)
                return marker

            marker_array.markers.append(create_points_marker(left_mask, 100, 0.0, 1.0, 0.0))
            marker_array.markers.append(create_points_marker(right_mask, 101, 1.0, 0.0, 0.0))
            
            self.gate_center = None
            
            if np.any(left_mask) and np.any(right_mask):
                left_idx = np.argmin(ranges[left_mask])
                left_r = ranges[left_mask][left_idx]
                left_a = angles[left_mask][left_idx]
                
                right_idx = np.argmin(ranges[right_mask])
                right_r = ranges[right_mask][right_idx]
                right_a = angles[right_mask][right_idx]
                
                lx = left_r * np.cos(left_a)
                ly = left_r * np.sin(left_a)
                
                rx = right_r * np.cos(right_a)
                ry = right_r * np.sin(right_a)
                
                cx = (lx + rx) / 2.0
                cy = (ly + ry) / 2.0
                
                self.gate_center = (cx, cy)
                
                gate_marker = Marker()
                gate_marker.header = scan_msg.header
                gate_marker.ns = "gate_center"
                gate_marker.id = 2
                gate_marker.type = Marker.SPHERE
                gate_marker.action = Marker.ADD
                gate_marker.pose.position.x = cx
                gate_marker.pose.position.y = cy
                gate_marker.pose.position.z = 0.2
                gate_marker.scale = Vector3(x=0.2, y=0.2, z=0.2)
                gate_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
                marker_array.markers.append(gate_marker)
                
            self.pub_debug_markers.publish(marker_array)

            if not self.gate_center:
                if self.state == 'APPROACH':
                    self.get_logger().info(
                        f"Въезд не найден. L={np.sum(left_mask)} R={np.sum(right_mask)}", 
                        throttle_duration_sec=1.0
                    )
        
        elif self.state == 'TUNNEL':
            # Центрирование по стенам
            right_wall_mask = (angles > 1.0) & (angles < 2.1) & (ranges < 2.0)
            left_wall_mask = (angles > -2.1) & (angles < -1.0) & (ranges < 2.0)
            
            self.left_wall_dist = None
            self.right_wall_dist = None
            
            if np.any(left_wall_mask):
                self.left_wall_dist = np.min(ranges[left_wall_mask])
                
            if np.any(right_wall_mask):
                self.right_wall_dist = np.min(ranges[right_wall_mask])
                
            marker_array = MarkerArray()
            
            def create_wall_marker(mask, id, r, g, b):
                marker = Marker()
                marker.header = scan_msg.header
                marker.ns = "tunnel_walls"
                marker.id = id
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
                marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                
                if np.any(mask):
                    valid_r = ranges[mask]
                    valid_a = angles[mask]
                    for r_val, a_val in zip(valid_r, valid_a):
                        p = Point()
                        p.x = r_val * np.cos(a_val)
                        p.y = r_val * np.sin(a_val)
                        p.z = 0.0
                        marker.points.append(p)
                return marker

            marker_array.markers.append(create_wall_marker(left_wall_mask, 10, 0.0, 1.0, 1.0))
            marker_array.markers.append(create_wall_marker(right_wall_mask, 11, 1.0, 0.0, 1.0))
            self.pub_debug_markers.publish(marker_array)

    def world_to_grid(self, x, y):
        if x < 0 or x >= self.grid_length:
            return None
        if y < -self.grid_width/2 or y >= self.grid_width/2:
            return None
            
        row = int(x / self.grid_resolution)
        col = int((y + self.grid_width/2) / self.grid_resolution)
        
        if row < 0 or row >= self.grid_h or col < 0 or col >= self.grid_w:
            return None
            
        return (row, col)

    def grid_to_world(self, row, col):
        x = (row + 0.5) * self.grid_resolution
        y = (col + 0.5) * self.grid_resolution - self.grid_width/2
        return (x, y)

    def build_occupancy_grid(self):
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        for point in self.left_lane_path:
            x, y = point
            y += self.lane_boundary_margin
            grid_pos = self.world_to_grid(x, y)
            if grid_pos:
                row, col = grid_pos
                self.occupancy_grid[row, col] = self.forbidden_cost
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                            self.occupancy_grid[r, c] = self.forbidden_cost

        for point in self.right_lane_path:
            x, y = point
            y -= self.lane_boundary_margin
            grid_pos = self.world_to_grid(x, y)
            if grid_pos:
                row, col = grid_pos
                self.occupancy_grid[row, col] = self.forbidden_cost
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                            self.occupancy_grid[r, c] = self.forbidden_cost
                            
        self.publish_debug_grid()

    def publish_debug_grid(self):
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "robot/base_link"
        
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_h
        grid_msg.info.height = self.grid_w
        grid_msg.info.origin.position.x = 0.0
        grid_msg.info.origin.position.y = -self.grid_width / 2.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        grid_t = self.occupancy_grid.T
        flat_grid = np.zeros_like(grid_t, dtype=np.int8)
        flat_grid[grid_t >= self.forbidden_cost] = 100
        
        grid_msg.data = flat_grid.flatten().tolist()
        self.pub_grid.publish(grid_msg)

    def find_path(self):
        lookahead_row = int(self.look_ahead_distance / self.grid_resolution)
        
        if lookahead_row >= self.grid_h:
            return None
            
        free_cols = []
        for col in range(self.grid_w):
            if self.occupancy_grid[lookahead_row, col] < self.forbidden_cost:
                free_cols.append(col)
                
        if not free_cols:
            lookahead_row = int(0.3 / self.grid_resolution)
            free_cols = []
            for col in range(self.grid_w):
                if self.occupancy_grid[lookahead_row, col] < self.forbidden_cost:
                    free_cols.append(col)
            
            if not free_cols:
                return None
        
        segments = []
        current_segment = [free_cols[0]]
        for i in range(1, len(free_cols)):
            if free_cols[i] == free_cols[i-1] + 1:
                current_segment.append(free_cols[i])
            else:
                segments.append(current_segment)
                current_segment = [free_cols[i]]
        segments.append(current_segment)
        
        center_col = self.grid_w // 2
        best_segment = min(segments, key=lambda s: abs((s[0]+s[-1])/2 - center_col))
        
        target_col = best_segment[len(best_segment)//2]
        
        return self.grid_to_world(lookahead_row, target_col)

    def control_loop(self):
        twist = Twist()
        
        if self.state == 'APPROACH':
            if self.gate_center:
                gx, gy = self.gate_center
                
                target_x = -gx
                target_y = -gy
                
                dist = np.sqrt(target_x**2 + target_y**2)
                
                twist.linear.x = self.speed
                angle = np.arctan2(target_y, target_x)
                twist.angular.z = angle * self.steering_gain
                
                if dist < 0.15: 
                    self.state = 'TUNNEL'
                    self.get_logger().info(f"Переход: APPROACH -> TUNNEL. Dist={dist:.2f}")
                    
            else:
                self.get_logger().info("Въезд не найден", throttle_duration_sec=1.0)
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                
        elif self.state == 'TUNNEL':
            twist.linear.x = self.speed
            
            if hasattr(self, 'left_wall_dist') and hasattr(self, 'right_wall_dist') and \
               self.left_wall_dist is not None and self.right_wall_dist is not None:
                
                critical_dist = 0.10
                
                if self.left_wall_dist < critical_dist:
                    twist.angular.z = -1.5
                    self.get_logger().warn(f"КРИТИЧЕСКИ БЛИЗКО К ЛЕВОЙ: {self.left_wall_dist:.2f}")
                elif self.right_wall_dist < critical_dist:
                    twist.angular.z = 1.5
                    self.get_logger().warn(f"КРИТИЧЕСКИ БЛИЗКО К ПРАВОЙ: {self.right_wall_dist:.2f}")
                else:
                    error = self.left_wall_dist - self.right_wall_dist
                    
                    kp = 1.5
                    kd = 0.5
                    
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    dt = current_time - self.last_time if hasattr(self, 'last_time') else 0.1
                    self.last_time = current_time
                    
                    derivative = (error - self.last_error) / dt if hasattr(self, 'last_error') else 0.0
                    self.last_error = error
                    
                    twist.angular.z = kp * error + kd * derivative
                    twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
                    
                    self.get_logger().info(f"TUNNEL: L={self.left_wall_dist:.2f}, R={self.right_wall_dist:.2f}, Err={error:.2f}", throttle_duration_sec=0.2)
            else:
                twist.angular.z = 0.0
                self.get_logger().info(f"Стены не видны! L={self.left_wall_dist}, R={self.right_wall_dist}", throttle_duration_sec=0.5)
            
        self.pub_cmd.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TunnelNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
