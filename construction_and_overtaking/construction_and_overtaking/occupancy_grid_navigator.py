#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, Bool, Header
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
import numpy as np
import time
import cv2


class OccupancyGridNavigator(Node):
    """Навигация с сеткой занятости для объезда конусов"""

    def __init__(self):
        super().__init__('occupancy_grid_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        self.sub_tunnel_entered = self.create_subscription(Bool, '/tunnel/entered', self.tunnel_entered_callback, 10)
        
        self.sub_left_path = self.create_subscription(Path, '/detect/lane_left_path', self.left_path_callback, 10)
        self.sub_right_path = self.create_subscription(Path, '/detect/lane_right_path', self.right_path_callback, 10)
        self.sub_pixel_counts = self.create_subscription(Point, '/detect/lane_pixel_counts', self.pixel_counts_callback, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_control_active = self.create_publisher(Bool, '/lane_control_active', 1)
        
        self.pub_grid = self.create_publisher(OccupancyGrid, '/debug/grid', 10)
        self.pub_path = self.create_publisher(Path, '/debug/path', 10)

        # Состояние
        self.left_distance = 999.0
        self.right_distance = 999.0
        self.white_pixels = 0.0
        self.yellow_pixels = 0.0
        self.current_path = []
        self.left_lane_path = []
        self.right_lane_path = []
        
        # Сетка
        self.grid_resolution = 0.02
        self.grid_width = 2.0
        self.grid_length = 2.0
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Робот
        self.robot_radius = 0.10
        self.safety_radius = 0.30
        self.min_passage_width = 0.15
        
        # Полоса
        self.lane_width_pixels = 450.0
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.05
        
        # Цены
        self.forbidden_cost = 1000.0
        self.obstacle_cost = 100.0
        
        # Управление
        self.speed = 0.12
        self.steering_gain = 1.0
        self.look_ahead_distance = 0.5
        
        # Вращение
        self.rotation_speed = 0.15
        self.rotating_mode = False
        
        self.lidar_received = False
        self.obstacle_count = 0
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(2.0, self.log_status)

        self.declare_parameter('enable_on_start', True)
        self.is_active = self.get_parameter('enable_on_start').value

        self.get_logger().info('=== Навигатор с сеткой занятости ===')

    def tunnel_entered_callback(self, msg):
        if msg.data and not self.is_active:
            self.is_active = True
            self.get_logger().info("Активирован!")

    def pixel_counts_callback(self, msg):
        self.white_pixels = msg.x
        self.yellow_pixels = msg.y

    def pixels_to_meters(self, pixel_distance):
        return pixel_distance / self.pixels_per_meter

    def get_lane_boundaries_in_meters(self):
        left_boundary = 999.0
        right_boundary = -999.0
        
        if self.left_distance < 999.0:
            left_boundary = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
        
        if self.right_distance < 999.0:
            right_boundary = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
            
        return left_boundary, right_boundary

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

    def is_passage_wide_enough(self, grid_pos, min_width_meters=None):
        if min_width_meters is None:
            min_width_meters = self.min_passage_width
            
        row, col = grid_pos
        min_width_cells = int(min_width_meters / self.grid_resolution)
        
        left_extent = 0
        for c in range(col - 1, -1, -1):
            if self.occupancy_grid[row, c] >= self.forbidden_cost:
                break
            left_extent += 1
            if left_extent >= min_width_cells // 2:
                break
        
        right_extent = 0
        for c in range(col + 1, self.grid_w):
            if self.occupancy_grid[row, c] >= self.forbidden_cost:
                break
            right_extent += 1
            if right_extent >= min_width_cells // 2:
                break
        
        total_width_cells = left_extent + 1 + right_extent
        return total_width_cells >= min_width_cells

    def try_find_path(self):
        goal_grid = None
        min_width_cells = int(self.min_passage_width / self.grid_resolution)
        
        start_scan_row = int(1.5 / self.grid_resolution)
        end_scan_row = int(0.5 / self.grid_resolution)
        
        for row in range(start_scan_row, end_scan_row, -1):
            if row >= self.grid_h:
                continue
            
            free_cols = []
            for col in range(self.grid_w):
                if self.occupancy_grid[row, col] < self.obstacle_cost:
                    free_cols.append(col)
            
            if len(free_cols) < min_width_cells:
                continue
            
            segments = []
            if not free_cols:
                continue
                
            current_segment = [free_cols[0]]
            for i in range(1, len(free_cols)):
                if free_cols[i] == free_cols[i-1] + 1:
                    current_segment.append(free_cols[i])
                else:
                    if len(current_segment) >= min_width_cells:
                        segments.append(current_segment)
                    current_segment = [free_cols[i]]
            
            if len(current_segment) >= min_width_cells:
                segments.append(current_segment)
            
            if not segments:
                continue
            
            best_segment = max(segments, key=len)
            goal_col = best_segment[len(best_segment)//2]
            goal_grid = (row, goal_col)
            
            if self.is_passage_wide_enough(goal_grid):
                break
            else:
                goal_grid = None
        
        if goal_grid is None:
            goal_grid = self.world_to_grid(0.5, 0.0)
        
        start_grid = self.world_to_grid(0.05, 0.0)
        
        if start_grid is None or goal_grid is None:
            return []
        
        return self.find_path_astar(start_grid, goal_grid)

    def build_occupancy_grid(self, lidar_points, stamp, frame_id):
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        
        # Левая полоса
        for point in self.left_lane_path:
            x, y = point
            y += self.lane_boundary_margin
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                self.occupancy_grid[row, col] = self.forbidden_cost
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                            self.occupancy_grid[r, c] = self.forbidden_cost

        # Правая полоса
        for point in self.right_lane_path:
            x, y = point
            y -= self.lane_boundary_margin
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                self.occupancy_grid[row, col] = self.forbidden_cost
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                            self.occupancy_grid[r, c] = self.forbidden_cost

        # Запасные прямые линии
        if not self.left_lane_path and left_boundary < 999.0:
             for row in range(self.grid_h):
                x, _ = self.grid_to_world(row, 0)
                if x > 1.0: continue
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    if y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost

        if not self.right_lane_path and right_boundary > -999.0:
             for row in range(self.grid_h):
                x, _ = self.grid_to_world(row, 0)
                if x > 1.0: continue
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    if y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
        
        # Препятствия с градиентом
        obstacle_map = np.ones((self.grid_h, self.grid_w), dtype=np.uint8)
        
        obstacle_points = 0
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                obstacle_map[row, col] = 0
                obstacle_points += 1
        
        obstacle_map[self.occupancy_grid >= self.forbidden_cost] = 0
        
        dist_grid = cv2.distanceTransform(obstacle_map, cv2.DIST_L2, 5)
        dist_m = dist_grid * self.grid_resolution
        
        lethal_mask = dist_m < self.robot_radius
        gradient_mask = (dist_m >= self.robot_radius) & (dist_m < self.safety_radius)
        
        self.occupancy_grid[lethal_mask] = np.maximum(self.occupancy_grid[lethal_mask], self.obstacle_cost)
        
        if np.any(gradient_mask):
            factor = (self.safety_radius - dist_m[gradient_mask]) / (self.safety_radius - self.robot_radius)
            gradient_cost = factor * self.obstacle_cost
            self.occupancy_grid[gradient_mask] = np.maximum(self.occupancy_grid[gradient_mask], gradient_cost)
        
        self.obstacle_count = obstacle_points
        self.publish_debug_grid(stamp, frame_id)

    def publish_debug_grid(self, stamp, frame_id):
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = rclpy.time.Time().to_msg()
        grid_msg.header.frame_id = frame_id
        
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
        flat_grid[(grid_t >= self.obstacle_cost) & (grid_t < self.forbidden_cost)] = 50
        
        grid_msg.data = flat_grid.flatten().tolist()
        self.pub_grid.publish(grid_msg)

    def find_path_astar(self, start_pos, goal_pos):
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        if self.occupancy_grid[goal_row, goal_col] >= self.forbidden_cost:
            return []
        
        if not self.is_passage_wide_enough(goal_pos):
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
                    
                    if not self.is_passage_wide_enough(neighbor):
                        continue
                    
                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    penalty = cell_cost * 0.1
                    tentative_g = g_score[current] + move_cost + penalty
                    
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
        
        simplified = world_path[::4]
        if len(world_path) > 0 and world_path[-1] not in simplified:
            simplified.append(world_path[-1])
        
        return simplified

    def lidar_callback(self, scan_msg):
        self.lidar_received = True
        
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
        
        x_robot = -x
        y_robot = -y
        
        mask = (x_robot > 0) & (x_robot < self.grid_length) & \
               (np.abs(y_robot) < self.grid_width/2) & (ranges < 10.0)
        
        lidar_points = np.column_stack([x_robot[mask], y_robot[mask]])
        self.last_frame_id = "robot/base_link"
        self.last_scan_time = scan_msg.header.stamp
        self.build_occupancy_grid(lidar_points, scan_msg.header.stamp, "robot/base_link")

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

    def log_status(self):
        obstacles_detected = self.check_obstacles()
        self.get_logger().info(
            f'[DEBUG] LiDAR: {"✓" if self.lidar_received else "✗"} | '
            f'Препятствия: {self.obstacle_count} | '
            f'check_obstacles(): {obstacles_detected} | '
            f'Режим: {"ВРАЩЕНИЕ" if self.rotating_mode else "НОРМА"}'
        )

    def find_lookahead_point(self):
        if len(self.current_path) == 0:
            return None
        
        best_point = None
        best_dist_diff = float('inf')
        
        for point in self.current_path:
            x, y = point
            dist = np.sqrt(x**2 + y**2)
            
            if x < 0.05:
                continue
            
            dist_diff = abs(dist - self.look_ahead_distance)
            if dist_diff < best_dist_diff:
                best_dist_diff = dist_diff
                best_point = point
        
        if best_point is None and len(self.current_path) > 0:
            best_point = self.current_path[-1]
        
        return best_point

    def check_obstacles(self):
        min_x, max_x = 0.0, 1.5
        min_y, max_y = -0.3, 0.3
        
        start_row = int(min_x / self.grid_resolution)
        end_row = int(max_x / self.grid_resolution)
        start_col = int((min_y + self.grid_width/2) / self.grid_resolution)
        end_col = int((max_y + self.grid_width/2) / self.grid_resolution)
        
        start_row = max(0, start_row)
        end_row = min(self.grid_h, end_row)
        start_col = max(0, start_col)
        end_col = min(self.grid_w, end_col)
        
        if start_row >= end_row or start_col >= end_col:
            return False
            
        roi = self.occupancy_grid[start_row:end_row, start_col:end_col]
        return np.any(roi == self.obstacle_cost)

    def control_loop(self):
        if not self.is_active:
            return

        if not self.check_obstacles():
            self.rotating_mode = False
            
            control_active = Bool()
            control_active.data = True
            self.pub_control_active.publish(control_active)
            
            self.current_path = []
            if hasattr(self, 'last_frame_id'):
                self.publish_debug_path([], self.last_frame_id)
            return

        path = self.try_find_path()
        
        if len(path) == 0:
            if not self.rotating_mode:
                self.get_logger().warn('[НЕТ ПУТИ] Включаем вращение', throttle_duration_sec=1.0)
                self.rotating_mode = True
            
            rotation_dir = -1.0 if self.yellow_pixels > self.white_pixels else 1.0
            
            control_active = Bool()
            control_active.data = False
            self.pub_control_active.publish(control_active)
            
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = rotation_dir * self.rotation_speed
            self.pub_cmd.publish(twist)
            
            return
        
        if self.rotating_mode:
            self.get_logger().info('[ПУТЬ НАЙДЕН]', throttle_duration_sec=0.5)
        
        self.rotating_mode = False
        self.current_path = path
        
        if hasattr(self, 'last_frame_id'):
            self.publish_debug_path(path, self.last_frame_id)
        
        self.follow_path()

    def publish_debug_path(self, path, frame_id):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rclpy.time.Time().to_msg()
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
            self.pub_cmd.publish(Twist())
            return
        
        target_x, target_y = target
        
        control_active = Bool()
        control_active.data = False
        self.pub_control_active.publish(control_active)
        
        twist = Twist()
        twist.linear.x = self.speed
        angle_to_target = np.arctan2(target_y, target_x)
        twist.angular.z = self.steering_gain * angle_to_target
        twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
        
        self.pub_cmd.publish(twist)


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
