#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from std_msgs.msg import Float64, UInt8, Bool, Header
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
import numpy as np
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2


class OccupancyGridNavigator(Node):
    """Ultra-Pure Occupancy Grid Navigation
    
    Grid -> Plan -> Follow
    """

    def __init__(self):
        super().__init__('occupancy_grid_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        self.sub_lane_state = self.create_subscription(UInt8, '/lane_detection_state', self.lane_state_callback, 10)
        
        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribe to lane paths
        self.sub_left_path = self.create_subscription(Path, '/detect/lane_left_path', self.left_path_callback, 10)
        self.sub_right_path = self.create_subscription(Path, '/detect/lane_right_path', self.right_path_callback, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/avoid_control', 10)
        self.pub_avoid_active = self.create_publisher(Bool, '/avoid_active', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)
        
        # Debug Publishers
        self.pub_grid = self.create_publisher(OccupancyGrid, '/debug/grid', 10)
        self.pub_path = self.create_publisher(Path, '/debug/path', 10)

        # State
        self.left_distance = 999.0
        self.right_distance = 999.0
        self.lane_state = 0
        self.current_path = []
        self.left_lane_path = []
        self.right_lane_path = []
        self.last_left_path_time = rclpy.time.Time(seconds=0)
        self.last_right_path_time = rclpy.time.Time(seconds=0)
        
        # Grid parameters
        self.grid_resolution = 0.02  # 2cm cells
        self.grid_width = 2.0
        
        # Grid X range (relative to robot)
        self.grid_min_x = -0.5
        self.grid_max_x = 1.5
        self.grid_length = self.grid_max_x - self.grid_min_x
        
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Robot parameters
        self.robot_radius = 0.12  # 12cm (increased safety margin)
        self.inflation_cells = int(np.ceil(self.robot_radius / self.grid_resolution))
        
        # Lane parameters
        self.lane_safety_margin = 0.10  # 10cm extra margin on each side of lane
        
        # Grid costs
        self.forbidden_cost = 100.0
        self.obstacle_cost = 100.0
        
        # Control parameters
        self.speed = 0.08  # Reduced from 0.10 for stability
        self.steering_gain = 1.5
        self.look_ahead_distance = 0.25  # Reduced from 0.3 for tighter control
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.log_status)

        self.get_logger().info('=== Grid Navigator (Stable Mode) ===' )
        self.get_logger().info(f'Speed: {self.speed}m/s | Lookahead: {self.look_ahead_distance}m')

    def world_to_grid(self, x, y):
        if x < self.grid_min_x or x >= self.grid_max_x:
            return None
        if y < -self.grid_width/2 or y >= self.grid_width/2:
            return None
            
        row = int((x - self.grid_min_x) / self.grid_resolution)
        col = int((y + self.grid_width/2) / self.grid_resolution)
        
        if row < 0 or row >= self.grid_h or col < 0 or col >= self.grid_w:
            return None
            
        return (row, col)

    def grid_to_world(self, row, col):
        x = (row + 0.5) * self.grid_resolution + self.grid_min_x
        y = (col + 0.5) * self.grid_resolution - self.grid_width/2
        return (x, y)

    def draw_thick_line_on_grid(self, p1, p2, thickness_cells=5):
        """Draw a thick line with safety margin"""
        grid_p1 = self.world_to_grid(p1[0], p1[1])
        grid_p2 = self.world_to_grid(p2[0], p2[1])
        
        if grid_p1 is None or grid_p2 is None:
            return

        r0, c0 = grid_p1
        r1, c1 = grid_p2
        
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        while True:
            # Draw thick point
            for d_r in range(-thickness_cells, thickness_cells + 1):
                for d_c in range(-thickness_cells, thickness_cells + 1):
                    rr, cc = r0 + d_r, c0 + d_c
                    if 0 <= rr < self.grid_h and 0 <= cc < self.grid_w:
                        self.occupancy_grid[rr, cc] = self.forbidden_cost

            if r0 == r1 and c0 == c1:
                break
            
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc

    def build_occupancy_grid(self, lidar_points, stamp, frame_id):
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Draw lanes from paths - NO TIMEOUT, always draw if available
        paths_to_draw = []
        
        if len(self.left_lane_path) > 0:
            paths_to_draw.append(self.left_lane_path)
        
        if len(self.right_lane_path) > 0:
            paths_to_draw.append(self.right_lane_path)

        for path_points in paths_to_draw:
            if len(path_points) < 2:
                continue
            
            # Connect consecutive points with THICK lines
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                
                # Reduced max distance to prevent connecting across gaps
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist < 0.15:  # Only connect if < 15cm apart
                    self.draw_thick_line_on_grid(p1, p2, thickness_cells=5)
        
        # Draw current lidar obstacles
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                for dr in range(-self.inflation_cells, self.inflation_cells + 1):
                    for dc in range(-self.inflation_cells, self.inflation_cells + 1):
                        if dr*dr + dc*dc <= self.inflation_cells*self.inflation_cells:
                            r, c = row + dr, col + dc
                            if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                                self.occupancy_grid[r, c] = self.obstacle_cost
        
        self.publish_debug_grid(stamp, frame_id)

    def publish_debug_grid(self, stamp, frame_id):
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = rclpy.time.Time(seconds=0).to_msg()
        grid_msg.header.frame_id = frame_id
        
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_h
        grid_msg.info.height = self.grid_w
        
        grid_msg.info.origin.position.x = self.grid_min_x
        grid_msg.info.origin.position.y = -self.grid_width / 2.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        grid_t = self.occupancy_grid.T
        flat_grid = np.zeros_like(grid_t, dtype=np.int8)
        flat_grid[grid_t >= self.forbidden_cost] = 100
        
        grid_msg.data = flat_grid.flatten().tolist()
        self.pub_grid.publish(grid_msg)

    def find_path_astar(self, start_pos, goal_pos):
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        if self.occupancy_grid[goal_row, goal_col] >= self.forbidden_cost:
            return []
        
        open_set = [(0, start_pos)]
        came_from = {}
        g_score = {start_pos: 0}
        closed_set = set()
        
        while open_set:
            open_set.sort(key=lambda x: x[0])
            current_f, current = open_set.pop(0)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            if current == goal_pos:
                return self.reconstruct_path(came_from, current)
            
            row, col = current
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    neighbor = (row + dr, col + dc)
                    nr, nc = neighbor
                    
                    if not (0 <= nr < self.grid_h and 0 <= nc < self.grid_w):
                        continue
                    
                    if neighbor in closed_set:
                        continue
                    
                    cell_cost = self.occupancy_grid[nr, nc]
                    
                    if cell_cost >= self.forbidden_cost:
                        continue
                    
                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    tentative_g = g_score[current] + move_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, goal_pos)
                        open_set.append((f, neighbor))
        
        return []

    def heuristic(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        world_path = []
        for row, col in path:
            x, y = self.grid_to_world(row, col)
            world_path.append((x, y))
        
        # Less aggressive simplification for smoother paths
        simplified = world_path[::3]
        if len(world_path) > 0 and world_path[-1] not in simplified:
            simplified.append(world_path[-1])
        
        return simplified

    def lidar_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')
        ranges[ranges < scan_msg.range_min] = float('inf')
        ranges[ranges > scan_msg.range_max] = float('inf')
        ranges[np.isnan(ranges)] = float('inf')

        angles = np.arange(
            scan_msg.angle_min, 
            scan_msg.angle_min + len(ranges) * scan_msg.angle_increment, 
            scan_msg.angle_increment
        )
        
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        elif len(angles) < len(ranges):
             angles = np.pad(angles, (0, len(ranges) - len(angles)), 'edge')

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Transform to base_link (180 deg rotation)
        x_robot = -x
        y_robot = -y
        
        # WIDER FOV: 120 degrees instead of 80
        angles_robot = np.arctan2(y_robot, x_robot)
        fov_limit = 120 * np.pi / 180
        mask_fov = np.abs(angles_robot) < fov_limit
        
        mask = (x_robot > self.grid_min_x) & (x_robot < self.grid_max_x) & \
               (np.abs(y_robot) < self.grid_width/2) & (ranges < 10.0) & mask_fov
        
        lidar_points = np.column_stack([x_robot[mask], y_robot[mask]])
        
        self.last_frame_id = "robot/base_link"
        self.last_scan_time = scan_msg.header.stamp
        self.build_occupancy_grid(lidar_points, scan_msg.header.stamp, "robot/base_link")

    def left_path_callback(self, msg):
        points = []
        for pose in msg.poses:
            points.append((pose.pose.position.x, pose.pose.position.y))
        self.left_lane_path = points
        self.last_left_path_time = rclpy.time.Time.from_msg(msg.header.stamp)

    def right_path_callback(self, msg):
        points = []
        for pose in msg.poses:
            points.append((pose.pose.position.x, pose.pose.position.y))
        self.right_lane_path = points
        self.last_right_path_time = rclpy.time.Time.from_msg(msg.header.stamp)

    def left_distance_callback(self, msg):
        self.left_distance = 999.0 if msg.data < 0 else msg.data

    def right_distance_callback(self, msg):
        self.right_distance = 999.0 if msg.data < 0 else msg.data

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def log_status(self):
        path_len = len(self.current_path)
        lane_paths = f"L:{len(self.left_lane_path)} R:{len(self.right_lane_path)}"
        
        target_info = ""
        if path_len > 0:
            target = self.find_lookahead_point()
            if target:
                tx, ty = target
                angle = np.arctan2(ty, tx) * 180 / np.pi
                target_info = f"→ ({tx:.2f},{ty:.2f}) {angle:.0f}°"
        
        self.get_logger().info(f'{lane_paths} | Path={path_len} {target_info}')

    def find_lookahead_point(self):
        if len(self.current_path) == 0:
            return None
        
        best_point = None
        best_dist_diff = float('inf')
        
        for point in self.current_path:
            x, y = point
            if x < 0.05:
                continue
            
            dist = np.sqrt(x**2 + y**2)
            dist_diff = abs(dist - self.look_ahead_distance)
            if dist_diff < best_dist_diff:
                best_dist_diff = dist_diff
                best_point = point
        
        if best_point is None and len(self.current_path) > 0:
            best_point = self.current_path[-1]
        
        return best_point

    def check_obstacles(self):
        min_x, max_x = 0.0, 1.0
        min_y, max_y = -0.25, 0.25
        
        start_row = int((min_x - self.grid_min_x) / self.grid_resolution)
        end_row = int((max_x - self.grid_min_x) / self.grid_resolution)
        
        start_col = int((min_y + self.grid_width/2) / self.grid_resolution)
        end_col = int((max_y + self.grid_width/2) / self.grid_resolution)
        
        start_row = max(0, start_row)
        end_row = min(self.grid_h, end_row)
        start_col = max(0, start_col)
        end_col = min(self.grid_w, end_col)
        
        if start_row >= end_row or start_col >= end_col:
            return False
            
        roi = self.occupancy_grid[start_row:end_row, start_col:end_col]
        return np.any(roi >= self.obstacle_cost)

    def control_loop(self):
        has_obstacles = self.check_obstacles()
        
        if not has_obstacles:
            avoid_active = Bool()
            avoid_active.data = False
            self.pub_avoid_active.publish(avoid_active)
            return

        # Find goal
        goal_grid = None
        start_scan_row = int((1.2 - self.grid_min_x) / self.grid_resolution)
        end_scan_row = int((0.4 - self.grid_min_x) / self.grid_resolution)
        
        for row in range(start_scan_row, end_scan_row, -1):
            if row >= self.grid_h or row < 0:
                continue
            
            free_cols = []
            for col in range(self.grid_w):
                if self.occupancy_grid[row, col] < self.obstacle_cost:
                    free_cols.append(col)
            
            if len(free_cols) > 8:  # Need decent width
                segments = []
                if not free_cols:
                    continue
                    
                current_segment = [free_cols[0]]
                for i in range(1, len(free_cols)):
                    if free_cols[i] == free_cols[i-1] + 1:
                        current_segment.append(free_cols[i])
                    else:
                        segments.append(current_segment)
                        current_segment = [free_cols[i]]
                segments.append(current_segment)
                
                best_segment = max(segments, key=len)
                goal_col = best_segment[len(best_segment)//2]
                goal_grid = (row, goal_col)
                break
        
        start_grid = self.world_to_grid(0.05, 0.0)
        
        if goal_grid is None:
            goal_grid = self.world_to_grid(0.5, 0.0)

        if start_grid is None or goal_grid is None:
            self.stop()
            return
        
        path = self.find_path_astar(start_grid, goal_grid)
        
        if len(path) == 0:
            self.get_logger().warn('[NO PATH]', throttle_duration_sec=1.0)
            self.stop()
            return
        
        self.current_path = path
        if hasattr(self, 'last_frame_id'):
            self.publish_debug_path(path, self.last_frame_id)
        self.follow_path()

    def publish_debug_path(self, path, frame_id):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rclpy.time.Time(seconds=0).to_msg()
        path_msg.header.frame_id = frame_id
        
        for x, y in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.pub_path.publish(path_msg)

    def follow_path(self):
        target = self.find_lookahead_point()
        
        if target is None:
            self.stop()
            return
        
        target_x, target_y = target
        
        twist = Twist()
        twist.linear.x = self.speed
        angle_to_target = np.arctan2(target_y, target_x)
        twist.angular.z = self.steering_gain * angle_to_target
        twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
        
        self.pub_cmd.publish(twist)
        
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        max_vel = Float64()
        max_vel.data = self.speed
        self.pub_max_vel.publish(max_vel)

    def stop(self):
        self.pub_cmd.publish(Twist())
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
