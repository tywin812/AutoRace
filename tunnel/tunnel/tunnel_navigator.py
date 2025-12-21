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
    """Tunnel Navigation Node
    
    Simplified navigation for tunnel:
    - Uses lane lines as walls
    - Extrapolates walls forward
    - Simple centering logic
    """

    def __init__(self):
        super().__init__('tunnel_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        
        # Subscribe to lane paths
        self.sub_left_path = self.create_subscription(Path, '/detect/lane_left_path', self.left_path_callback, 10)
        self.sub_right_path = self.create_subscription(Path, '/detect/lane_right_path', self.right_path_callback, 10)
        self.sub_pixel_counts = self.create_subscription(Point, '/detect/lane_pixel_counts', self.pixel_counts_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10) # Direct control for tunnel
        self.pub_tunnel_entered = self.create_publisher(Bool, '/tunnel/entered', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)
        
        # Debug Publishers
        self.pub_grid = self.create_publisher(OccupancyGrid, '/debug/grid', 10)
        self.pub_path = self.create_publisher(Path, '/debug/path', 10)
        self.pub_traversed_path = self.create_publisher(Path, '/tunnel/traversed_path', 10)
        self.pub_debug_markers = self.create_publisher(MarkerArray, '/tunnel/debug_markers', 10)

        # State
        self.traversed_path = Path()
        self.left_distance = 999.0
        self.right_distance = 999.0
        self.white_pixels = 0.0
        self.yellow_pixels = 0.0
        self.current_path = []
        self.left_lane_path = []
        self.right_lane_path = []
        
        # Grid parameters
        self.grid_resolution = 0.02  # 2cm cells
        self.grid_width = 2.0
        self.grid_length = 2.0
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Robot parameters
        self.robot_radius = 0.10
        self.safety_radius = 0.30
        
        # Lane parameters
        self.lane_boundary_margin = 0.10 # 10cm margin
        
        # Grid costs
        self.forbidden_cost = 1000.0
        self.obstacle_cost = 100.0
        
        # Control parameters
        self.speed = 0.08 # Very slow for safety
        self.steering_gain = 1.0
        self.look_ahead_distance = 0.6
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.state = 'APPROACH' # APPROACH, TUNNEL
        self.gate_center = None

        self.get_logger().info('=== Tunnel Navigator Started ===')

    def odom_callback(self, msg):
        # Append pose to path
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        
        self.traversed_path.header = msg.header
        self.traversed_path.poses.append(pose)
        
        # Limit path length to avoid memory issues (e.g. last 1000 points)
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
        ranges[np.isnan(ranges)] = float('inf') # Handle NaNs
        ranges[ranges == 0] = float('inf')
        ranges[np.isneginf(ranges)] = float('inf') # Handle -inf
        
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        # Normalize angles to [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        
        if self.state == 'APPROACH':
            # --- GATE DETECTION LOGIC ---
            
            # Debug: Print stats about what we see
            valid_ranges = ranges[ranges < 10.0] # Filter out infs for stats
            if len(valid_ranges) > 0:
                 self.get_logger().info(
                    f"Lidar stats: Min Range={np.min(valid_ranges):.2f}, Max Range={np.max(valid_ranges):.2f}. "
                    f"Angles: {np.min(angles):.2f} to {np.max(angles):.2f}",
                    throttle_duration_sec=2.0
                )
            
            # User indicated Front/Back confusion. Switching to look "Behind" (angles near +/- PI)
            # This assumes 0 is Back and +/- PI is Front
            
            # Left sector: +2.0 rad to +PI
            left_mask = (angles > 2.0) & (ranges < 3.0)
            # Right sector: -PI to -2.0 rad
            right_mask = (angles < -2.0) & (ranges < 3.0)
            
            # Visualization
            marker_array = MarkerArray()
            
            # Helper to create sector boundary markers
            def create_sector_marker(id, min_angle, max_angle, r, g, b):
                marker = Marker()
                marker.header = scan_msg.header
                marker.ns = "gate_sectors"
                marker.id = id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale = Vector3(x=0.05, y=0.0, z=0.0) # Line width
                marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                
                # Draw a "pie slice" or just the boundary lines
                # Center
                p0 = Point(x=0.0, y=0.0, z=0.0)
                
                # Max range for visualization
                vis_range = 2.0
                
                # Min angle line
                p1 = Point()
                p1.x = vis_range * np.cos(min_angle)
                p1.y = vis_range * np.sin(min_angle)
                
                # Max angle line
                p2 = Point()
                p2.x = vis_range * np.cos(max_angle)
                p2.y = vis_range * np.sin(max_angle)
                
                # Arc (simplified as lines)
                marker.points.append(p0)
                marker.points.append(p1)
                marker.points.append(p2)
                marker.points.append(p0)
                
                return marker

            # Visualize Left Sector (Green boundary) - looking "back-left"
            marker_array.markers.append(create_sector_marker(0, 2.0, 3.14, 0.0, 1.0, 0.0))
            # Visualize Right Sector (Red boundary) - looking "back-right"
            marker_array.markers.append(create_sector_marker(1, -3.14, -2.0, 1.0, 0.0, 0.0))
            
            # Helper to create points marker (only for valid points)
            def create_points_marker(mask, id, r, g, b):
                marker = Marker()
                marker.header = scan_msg.header
                marker.ns = "gate_search"
                marker.id = id
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
                marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                
                # If mask is None, use all valid points
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

            # marker_array.markers.append(create_points_marker(None, 10, 1.0, 1.0, 1.0)) # White for ALL points - REMOVED
            marker_array.markers.append(create_points_marker(left_mask, 100, 0.0, 1.0, 0.0)) # Green for Left points
            marker_array.markers.append(create_points_marker(right_mask, 101, 1.0, 0.0, 0.0)) # Red for Right points
            
            self.gate_center = None
            
            if np.any(left_mask) and np.any(right_mask):
                # Find closest points in each sector
                left_idx = np.argmin(ranges[left_mask])
                left_r = ranges[left_mask][left_idx]
                left_a = angles[left_mask][left_idx]
                
                right_idx = np.argmin(ranges[right_mask])
                right_r = ranges[right_mask][right_idx]
                right_a = angles[right_mask][right_idx]
                
                # Convert to Cartesian
                lx = left_r * np.cos(left_a)
                ly = left_r * np.sin(left_a)
                
                rx = right_r * np.cos(right_a)
                ry = right_r * np.sin(right_a)
                
                # Calculate midpoint
                cx = (lx + rx) / 2.0
                cy = (ly + ry) / 2.0
                
                self.gate_center = (cx, cy)
                
                # Visualize Gate Center
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
                gate_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) # Blue
                marker_array.markers.append(gate_marker)
                
            self.pub_debug_markers.publish(marker_array)

            if not self.gate_center:
                # Debug info if gate not found
                if self.state == 'APPROACH':
                    self.get_logger().info(
                        f"Gate not found. Raw Ang: [{scan_msg.angle_min:.2f}, {scan_msg.angle_max:.2f}]. "
                        f"Valid pts: L={np.sum(left_mask)} R={np.sum(right_mask)}", 
                        throttle_duration_sec=1.0
                    )
        
        elif self.state == 'TUNNEL':
            # --- WALL CENTERING LOGIC ---
            # 0 is Back. +/- PI is Front.
            # Right is +PI/2 (+1.57). Left is -PI/2 (-1.57).
            
            # Define sectors for side walls (widened to catch walls even if rotated)
            # Right: +1.0 to +2.1 rad
            # Left: -2.1 to -1.0 rad
            
            right_wall_mask = (angles > 1.0) & (angles < 2.1) & (ranges < 2.0)
            left_wall_mask = (angles > -2.1) & (angles < -1.0) & (ranges < 2.0)
            
            self.left_wall_dist = None
            self.right_wall_dist = None
            
            if np.any(left_wall_mask):
                # Use MIN distance instead of MEAN to find the closest point on the wall
                self.left_wall_dist = np.min(ranges[left_wall_mask])
                
            if np.any(right_wall_mask):
                # Use MIN distance instead of MEAN
                self.right_wall_dist = np.min(ranges[right_wall_mask])
                
            # Debug visualization for walls
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

            marker_array.markers.append(create_wall_marker(left_wall_mask, 10, 0.0, 1.0, 1.0)) # Cyan for Left Wall
            marker_array.markers.append(create_wall_marker(right_wall_mask, 11, 1.0, 0.0, 1.0)) # Magenta for Right Wall
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
        
        # Draw lanes
        for point in self.left_lane_path:
            x, y = point
            y += self.lane_boundary_margin
            grid_pos = self.world_to_grid(x, y)
            if grid_pos:
                row, col = grid_pos
                self.occupancy_grid[row, col] = self.forbidden_cost
                # Thicken
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
                # Thicken
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
        # Simple centering logic: Find center of free space at lookahead distance
        lookahead_row = int(self.look_ahead_distance / self.grid_resolution)
        
        if lookahead_row >= self.grid_h:
            return None
            
        # Find free cells in lookahead row
        free_cols = []
        for col in range(self.grid_w):
            if self.occupancy_grid[lookahead_row, col] < self.forbidden_cost:
                free_cols.append(col)
                
        if not free_cols:
            # Try closer
            lookahead_row = int(0.3 / self.grid_resolution)
            free_cols = []
            for col in range(self.grid_w):
                if self.occupancy_grid[lookahead_row, col] < self.forbidden_cost:
                    free_cols.append(col)
            
            if not free_cols:
                return None
        
        # Find center of largest segment
        segments = []
        current_segment = [free_cols[0]]
        for i in range(1, len(free_cols)):
            if free_cols[i] == free_cols[i-1] + 1:
                current_segment.append(free_cols[i])
            else:
                segments.append(current_segment)
                current_segment = [free_cols[i]]
        segments.append(current_segment)
        
        # Pick segment closest to center
        center_col = self.grid_w // 2
        best_segment = min(segments, key=lambda s: abs((s[0]+s[-1])/2 - center_col))
        
        target_col = best_segment[len(best_segment)//2]
        
        return self.grid_to_world(lookahead_row, target_col)

    def control_loop(self):
        twist = Twist()
        
        if self.state == 'APPROACH':
            if self.gate_center:
                gx, gy = self.gate_center
                
                # Transform to base_link (assuming lidar is rotated 180 relative to robot front)
                # We are detecting gate at angles +/- PI, which corresponds to -X in Lidar frame.
                # This is +X in Robot frame.
                target_x = -gx
                target_y = -gy
                
                dist = np.sqrt(target_x**2 + target_y**2)
                
                # Drive to gate
                twist.linear.x = self.speed
                angle = np.arctan2(target_y, target_x)
                twist.angular.z = angle * self.steering_gain
                
                # Transition condition: Close to gate (using distance, not raw X)
                if dist < 0.15: 
                    self.state = 'TUNNEL'
                    self.get_logger().info(f"TRANSITION: APPROACH -> TUNNEL. Dist={dist:.2f}. Switching to Wall Centering.")
                    
            else:
                self.get_logger().info("APPROACH: No gate detected", throttle_duration_sec=1.0)
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                
        elif self.state == 'TUNNEL':
            # Simple Wall Centering
            # Error = Left - Right
            # If Left > Right (positive error) -> Turn Left (positive angular.z)
            # If Right > Left (negative error) -> Turn Right (negative angular.z)
            
            twist.linear.x = self.speed
            
            if hasattr(self, 'left_wall_dist') and hasattr(self, 'right_wall_dist') and \
               self.left_wall_dist is not None and self.right_wall_dist is not None:
                
                # Emergency Wall Avoidance
                # If too close to a wall, steer away aggressively
                critical_dist = 0.10
                
                if self.left_wall_dist < critical_dist:
                    twist.angular.z = -1.5 # Turn Right Hard
                    self.get_logger().warn(f"CRITICAL LEFT: {self.left_wall_dist:.2f}. Turning Right!")
                elif self.right_wall_dist < critical_dist:
                    twist.angular.z = 1.5 # Turn Left Hard
                    self.get_logger().warn(f"CRITICAL RIGHT: {self.right_wall_dist:.2f}. Turning Left!")
                else:
                    # Normal PID Control
                    error = self.left_wall_dist - self.right_wall_dist
                    
                    # PD-Controller
                    kp = 1.5 # Increased from 0.5
                    kd = 0.5 # Increased from 0.1
                    
                    # Calculate derivative
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    dt = current_time - self.last_time if hasattr(self, 'last_time') else 0.1
                    self.last_time = current_time
                    
                    derivative = (error - self.last_error) / dt if hasattr(self, 'last_error') else 0.0
                    self.last_error = error
                    
                    twist.angular.z = kp * error + kd * derivative
                    
                    # Limit steering
                    twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
                    
                    self.get_logger().info(f"TUNNEL: L={self.left_wall_dist:.2f}, R={self.right_wall_dist:.2f}, Err={error:.2f}, Steer={twist.angular.z:.2f}", throttle_duration_sec=0.2)
            else:
                # Blind forward or keep previous steering?
                # Better to go straight if lost
                twist.angular.z = 0.0
                self.get_logger().info(f"TUNNEL: Lost walls! L={self.left_wall_dist}, R={self.right_wall_dist}", throttle_duration_sec=0.5)
            
        self.pub_cmd.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TunnelNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
