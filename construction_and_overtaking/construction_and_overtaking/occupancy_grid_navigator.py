#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
import numpy as np


class OccupancyGridNavigator(Node):
    """Ultra-Pure Occupancy Grid Navigation
    
    Simplest possible approach:
    1. Build occupancy grid from LiDAR + lane boundaries
    2. Plan path with A* every cycle
    3. Follow the path
    
    No checks, no optimizations, no reactive logic.
    Just: Grid → Plan → Follow
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
        self.left_distance = 999.0  # pixels
        self.right_distance = 999.0  # pixels
        self.lane_state = 0
        self.current_path = []  # Current planned path
        self.path_waypoint_index = 0
        
        # Grid parameters
        self.grid_resolution = 0.05  # 5cm per cell
        self.grid_width = 2.0  # 2m total width
        self.grid_length = 2.0  # 2m look ahead
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Lane parameters
        self.lane_width_pixels = 600.0
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.08  # meters
        
        # Grid costs
        self.forbidden_cost = 1000.0  # Behind lane lines - CANNOT pass
        self.obstacle_cost = 100.0    # Obstacles - expensive
        
        # Control parameters
        self.speed = 0.18
        self.steering_gain = 2.8
        self.waypoint_reached_threshold = 0.1  # meters
        
        # Control timer - plan and follow every cycle
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.log_status)

        self.get_logger().info('=== Ultra-Pure Occupancy Grid Navigator ===')
        self.get_logger().info(f'Grid: {self.grid_w}x{self.grid_h} @ {self.grid_resolution}m')
        self.get_logger().info('Strategy: Always plan with A*')

    def pixels_to_meters(self, pixel_distance):
        return pixel_distance / self.pixels_per_meter

    def get_lane_boundaries_in_meters(self):
        """Get lane boundaries in meters"""
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
        """World (meters) -> Grid (row, col)"""
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
        """Grid (row, col) -> World (meters)"""
        x = (row + 0.5) * self.grid_resolution
        y = (col + 0.5) * self.grid_resolution - self.grid_width/2
        return (x, y)

    def build_occupancy_grid(self, lidar_points):
        """Build grid: forbidden zones + obstacles"""
        # Reset
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Mark forbidden zones (behind lane lines)
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        
        if self.lane_state > 0:
            for row in range(self.grid_h):
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    
                    # Beyond left line
                    if left_boundary < 999.0 and y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
                    
                    # Beyond right line
                    elif right_boundary > -999.0 and y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
        
        # Mark obstacles from LiDAR
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                
                # Don't overwrite forbidden zones
                if self.occupancy_grid[row, col] < self.forbidden_cost:
                    # Inflate (3x3)
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
        """A* pathfinding"""
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        # Check goal validity
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
                    
                    # Skip forbidden cells
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
        """Reconstruct and simplify path"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        # Convert to world coords
        world_path = []
        for row, col in path:
            x, y = self.grid_to_world(row, col)
            world_path.append((x, y))
        
        # Simplify: every 3rd point
        simplified = world_path[::3]
        if len(world_path) > 0 and world_path[-1] not in simplified:
            simplified.append(world_path[-1])
        
        return simplified

    def lidar_callback(self, scan_msg):
        """Process LiDAR"""
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
        
        self.get_logger().info(
            f'Lanes: L={left_b:.2f}m R={right_b:.2f}m | '
            f'Path: {path_len} waypoints | '
            f'Waypoint: {self.path_waypoint_index}/{path_len}'
        )

    def control_loop(self):
        """Main loop: Always plan, always follow"""
        
        # Step 1: Plan path from current position to goal
        start_grid = self.world_to_grid(0.05, 0.0)  # Just ahead
        goal_grid = self.world_to_grid(self.grid_length - 0.2, 0.0)  # Far ahead, centered
        
        if start_grid is None or goal_grid is None:
            self.stop()
            return
        
        # Step 2: Run A* 
        path = self.find_path_astar(start_grid, goal_grid)
        
        if len(path) == 0:
            # No path found -> STOP (blocked by forbidden zones or obstacles)
            self.get_logger().warn('[GRID] No path! STOPPING.', throttle_duration_sec=1.0)
            self.stop()
            return
        
        # Step 3: Update current path
        self.current_path = path
        self.path_waypoint_index = 0  # Always start from first waypoint
        
        # Step 4: Follow path
        self.follow_path()

    def follow_path(self):
        """Follow the planned path"""
        if self.path_waypoint_index >= len(self.current_path):
            # Shouldn't happen since we replan every cycle
            self.stop()
            return
        
        # Get target waypoint
        target_x, target_y = self.current_path[self.path_waypoint_index]
        
        # Check if waypoint reached
        dist_to_waypoint = np.sqrt(target_x**2 + target_y**2)
        if dist_to_waypoint < self.waypoint_reached_threshold:
            # Move to next waypoint
            self.path_waypoint_index += 1
            if self.path_waypoint_index >= len(self.current_path):
                # Path completed, but we'll replan next cycle anyway
                pass
            return
        
        # Pure pursuit control
        twist = Twist()
        twist.linear.x = self.speed
        
        angle_to_target = np.arctan2(target_y, target_x)
        twist.angular.z = self.steering_gain * angle_to_target
        twist.angular.z = np.clip(twist.angular.z, -1.5, 1.5)
        
        self.pub_cmd.publish(twist)
        
        # Publish avoid active (always true when following path)
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        # Publish speed
        max_vel = Float64()
        max_vel.data = self.speed
        self.pub_max_vel.publish(max_vel)

    def stop(self):
        """Emergency stop"""
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
