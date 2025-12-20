#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
import numpy as np


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

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/avoid_control', 10)
        self.pub_avoid_active = self.create_publisher(Bool, '/avoid_active', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)

        # State
        self.left_distance = 999.0
        self.right_distance = 999.0
        self.lane_state = 0
        self.current_path = []
        
        # Grid parameters
        self.grid_resolution = 0.05
        self.grid_width = 2.0
        self.grid_length = 2.0
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Lane parameters
        self.lane_width_pixels = 600.0
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.08
        
        # Grid costs
        self.forbidden_cost = 1000.0
        self.obstacle_cost = 100.0
        
        # Control parameters
        self.speed = 0.18
        self.steering_gain = 3.0
        self.look_ahead_distance = 0.4
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.log_status)  # More frequent

        self.get_logger().info('=== Grid Navigator ===' )
        self.get_logger().info(f'Publishing to: /avoid_control, /avoid_active')

    def pixels_to_meters(self, pixel_distance):
        return pixel_distance / self.pixels_per_meter

    def get_lane_boundaries_in_meters(self):
        if self.left_distance < 999.0:
            left_boundary = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
        else:
            left_boundary = 999.0
            
        if self.right_distance < 999.0:
            right_boundary = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
        else:
            right_boundary = -999.0
            
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

    def build_occupancy_grid(self, lidar_points):
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        
        if self.lane_state > 0:
            for row in range(self.grid_h):
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    
                    if left_boundary < 999.0 and y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
                    elif right_boundary > -999.0 and y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
        
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                
                if self.occupancy_grid[row, col] < self.forbidden_cost:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            r, c = row + dr, col + dc
                            if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                                if self.occupancy_grid[r, c] < self.forbidden_cost:
                                    self.occupancy_grid[r, c] = max(
                                        self.occupancy_grid[r, c],
                                        self.obstacle_cost
                                    )

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
                    tentative_g = g_score[current] + move_cost + cell_cost
                    
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
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')
        ranges[ranges == float('-inf')] = float('inf')
        ranges[np.isnan(ranges)] = float('inf')

        ranges = np.roll(ranges, len(ranges)//2)
        angles = np.linspace(np.pi, 3*np.pi, len(ranges), endpoint=False)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        mask = (x > 0) & (x < self.grid_length) & \
               (np.abs(y) < self.grid_width/2) & (ranges < 10.0)
        
        lidar_points = np.column_stack([x[mask], y[mask]])
        self.build_occupancy_grid(lidar_points)

    def left_distance_callback(self, msg):
        self.left_distance = 999.0 if msg.data < 0 else msg.data

    def right_distance_callback(self, msg):
        self.right_distance = 999.0 if msg.data < 0 else msg.data

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def log_status(self):
        left_b, right_b = self.get_lane_boundaries_in_meters()
        path_len = len(self.current_path)
        
        target_info = ""
        if path_len > 0:
            target = self.find_lookahead_point()
            if target:
                tx, ty = target
                dist = np.sqrt(tx**2 + ty**2)
                angle = np.arctan2(ty, tx) * 180 / np.pi
                angular_z = self.steering_gain * np.arctan2(ty, tx)
                target_info = f"Target: ({tx:.2f},{ty:.2f}) {dist:.2f}m {angle:.0f}° -> ω={angular_z:.2f}"
        
        self.get_logger().info(
            f'L={left_b:.2f} R={right_b:.2f} | Path={path_len} | {target_info}'
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

    def control_loop(self):
        start_grid = self.world_to_grid(0.05, 0.0)
        goal_grid = self.world_to_grid(self.grid_length - 0.2, 0.0)
        
        if start_grid is None or goal_grid is None:
            self.stop()
            return
        
        path = self.find_path_astar(start_grid, goal_grid)
        
        if len(path) == 0:
            self.get_logger().warn('[NO PATH]', throttle_duration_sec=1.0)
            self.stop()
            return
        
        self.current_path = path
        self.follow_path()

    def follow_path(self):
        target = self.find_lookahead_point()
        
        if target is None:
            self.stop()
            return
        
        target_x, target_y = target
        
        # Calculate control
        twist = Twist()
        twist.linear.x = self.speed
        angle_to_target = np.arctan2(target_y, target_x)
        twist.angular.z = self.steering_gain * angle_to_target
        twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
        
        # Publish
        self.pub_cmd.publish(twist)
        
        # Debug log (throttled)
        self.get_logger().debug(
            f'CMD: v={twist.linear.x:.2f} ω={twist.angular.z:.2f} | '
            f'Target: ({target_x:.2f},{target_y:.2f})',
            throttle_duration_sec=0.5
        )
        
        # Always active
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        # Speed
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
