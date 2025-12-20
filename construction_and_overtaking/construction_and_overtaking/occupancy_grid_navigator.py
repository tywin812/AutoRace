#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
import numpy as np


class OccupancyGridNavigator(Node):
    """Pure Occupancy Grid Navigation with A* Path Planning
    
    Core concept:
    1. Build occupancy grid from LiDAR (obstacles) and camera (forbidden zones)
    2. Always plan path through free cells
    3. If path exists -> follow it
    4. If no path -> stop
    
    No reactive checks, no corridor detection - just grid + planning!
    """
    
    STATE_LANE_FOLLOWING = 0
    STATE_PATH_FOLLOWING = 1
    STATE_RETURNING = 2

    def __init__(self):
        super().__init__('occupancy_grid_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        self.sub_lane_state = self.create_subscription(UInt8, '/lane_detection_state', self.lane_state_callback, 10)

        # Publishers
        self.pub_avoid_cmd = self.create_publisher(Twist, '/avoid_control', 10)
        self.pub_avoid_active = self.create_publisher(Bool, '/avoid_active', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)

        # State variables
        self.state = self.STATE_LANE_FOLLOWING
        self.left_distance = 999.0  # pixels
        self.right_distance = 999.0  # pixels
        self.lane_state = 0
        self.current_path = []  # List of (x, y) waypoints in meters
        self.path_index = 0
        
        # Grid parameters
        self.grid_resolution = 0.05  # meters per cell (5cm)
        self.grid_width = 2.0  # meters (1m left + 1m right)
        self.grid_length = 2.0  # meters (look ahead)
        self.grid_w = int(self.grid_width / self.grid_resolution)  # cells
        self.grid_h = int(self.grid_length / self.grid_resolution)  # cells
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Lane parameters
        self.lane_width_pixels = 600.0
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.08  # meters
        
        # Grid costs
        self.forbidden_cost = 1000.0  # cells behind lane lines - CANNOT pass
        self.obstacle_cost = 100.0    # cells with obstacles - expensive to pass
        self.free_cost = 0.0          # free cells
        
        # Control parameters
        self.path_following_speed = 0.15
        self.lane_following_speed = 0.22
        self.waypoint_threshold = 0.12  # meters - waypoint reached
        self.steering_gain = 2.5
        
        # Planning parameters
        self.path_check_distance = 0.8  # meters - how far ahead to check
        self.min_clear_width = 0.25  # meters - minimum width for "clear path"
        
        # Timers
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.log_status)

        self.get_logger().info('=== Pure Occupancy Grid Navigator ===')
        self.get_logger().info(f'Grid: {self.grid_w}x{self.grid_h} cells @ {self.grid_resolution}m')
        self.get_logger().info(f'Area: {self.grid_length}m x {self.grid_width}m')

    def pixels_to_meters(self, pixel_distance):
        """Convert pixel distance from camera to meters"""
        return pixel_distance / self.pixels_per_meter

    def get_lane_boundaries_in_meters(self):
        """Get left and right lane boundaries in robot frame (meters)
        Returns: (left_boundary_y, right_boundary_y)
        """
        if self.left_distance < 999.0:
            left_boundary_m = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
        else:
            left_boundary_m = 999.0
            
        if self.right_distance < 999.0:
            right_boundary_m = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
        else:
            right_boundary_m = -999.0
            
        return left_boundary_m, right_boundary_m

    def world_to_grid(self, x, y):
        """Convert world coordinates (meters) to grid indices
        Robot is at bottom center of grid
        Returns: (row, col) or None if out of bounds
        """
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
        """Convert grid indices to world coordinates"""
        x = (row + 0.5) * self.grid_resolution
        y = (col + 0.5) * self.grid_resolution - self.grid_width/2
        return (x, y)

    def build_occupancy_grid(self, lidar_points):
        """Build occupancy grid from LiDAR and lane boundaries
        
        Priority:
        1. Mark forbidden zones (behind lines) - HIGHEST priority
        2. Mark obstacles from LiDAR
        """
        # Reset grid
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Step 1: Mark forbidden zones (behind lane lines)
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        
        if self.lane_state > 0:
            for row in range(self.grid_h):
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    
                    # Beyond left line -> FORBIDDEN
                    if left_boundary < 999.0 and y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
                    
                    # Beyond right line -> FORBIDDEN
                    elif right_boundary > -999.0 and y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
        
        # Step 2: Mark obstacles from LiDAR (don't overwrite forbidden zones)
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                
                # Only mark if not already forbidden
                if self.occupancy_grid[row, col] < self.forbidden_cost:
                    # Inflate obstacle (3x3)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            r, c = row + dr, col + dc
                            if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                                if self.occupancy_grid[r, c] < self.forbidden_cost:
                                    self.occupancy_grid[r, c] = max(
                                        self.occupancy_grid[r, c], 
                                        self.obstacle_cost
                                    )

    def is_straight_path_clear(self):
        """Check if straight path ahead is clear on the grid
        
        Returns: True if robot can go straight, False if needs to plan detour
        """
        # Check a narrow vertical strip in the center of the grid
        center_col = self.grid_w // 2
        strip_width = int(self.min_clear_width / self.grid_resolution)
        
        check_rows = int(self.path_check_distance / self.grid_resolution)
        check_rows = min(check_rows, self.grid_h)
        
        # Check center strip for obstacles or forbidden zones
        for row in range(check_rows):
            for dc in range(-strip_width, strip_width + 1):
                col = center_col + dc
                if 0 <= col < self.grid_w:
                    cost = self.occupancy_grid[row, col]
                    # If any cell is blocked -> path not clear
                    if cost >= self.obstacle_cost:
                        return False
        
        return True

    def find_path_astar(self, start_pos, goal_pos):
        """A* pathfinding on occupancy grid
        
        Returns: List of (x, y) waypoints or empty list if no path
        """
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        # Check if goal is valid
        if self.occupancy_grid[goal_row, goal_col] >= self.forbidden_cost:
            return []
        
        # Priority queue: (f_score, (row, col))
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
            
            # Goal reached
            if current == goal_pos:
                return self.reconstruct_path(came_from, current)
            
            # Check neighbors (8-connected)
            row, col = current
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    neighbor = (row + dr, col + dc)
                    nr, nc = neighbor
                    
                    # Bounds check
                    if not (0 <= nr < self.grid_h and 0 <= nc < self.grid_w):
                        continue
                    
                    if neighbor in closed_set:
                        continue
                    
                    # Get cell cost
                    cell_cost = self.occupancy_grid[nr, nc]
                    
                    # SKIP forbidden cells
                    if cell_cost >= self.forbidden_cost:
                        continue
                    
                    # Movement cost
                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    tentative_g = g_score[current] + move_cost + cell_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, goal_pos)
                        open_set.append((f, neighbor))
        
        return []  # No path found

    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path and convert to world coordinates"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        # Convert to world coordinates
        world_path = []
        for row, col in path:
            x, y = self.grid_to_world(row, col)
            world_path.append((x, y))
        
        # Simplify: keep every 3rd waypoint
        simplified = world_path[::3]
        if len(world_path) > 0 and world_path[-1] not in simplified:
            simplified.append(world_path[-1])
        
        return simplified

    def lidar_callback(self, scan_msg):
        """Process LiDAR scan"""
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')
        ranges[ranges == float('-inf')] = float('inf')
        ranges[np.isnan(ranges)] = float('inf')

        # Rotate to front
        ranges = np.roll(ranges, len(ranges)//2)
        angles = np.linspace(np.pi, 3*np.pi, len(ranges), endpoint=False)

        # Cartesian conversion
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Filter to grid area
        mask = (x > 0) & (x < self.grid_length) & \
               (np.abs(y) < self.grid_width/2) & (ranges < 10.0)
        
        lidar_points = np.column_stack([x[mask], y[mask]])
        
        # Build grid
        self.build_occupancy_grid(lidar_points)

    def left_distance_callback(self, msg):
        self.left_distance = 999.0 if msg.data < 0 else msg.data

    def right_distance_callback(self, msg):
        self.right_distance = 999.0 if msg.data < 0 else msg.data

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def log_status(self):
        """Log status"""
        state_names = {0: "LANE_FOLLOW", 1: "PATH_FOLLOW", 2: "RETURNING"}
        state_name = state_names.get(self.state, "UNKNOWN")
        
        left_b, right_b = self.get_lane_boundaries_in_meters()
        straight_clear = self.is_straight_path_clear()
        
        self.get_logger().info(
            f'[{state_name}] Lanes: L={left_b:.2f}m R={right_b:.2f}m | '
            f'Straight: {"CLEAR" if straight_clear else "BLOCKED"} | '
            f'Path: {len(self.current_path)} pts'
        )

    def control_loop(self):
        """Main control loop - pure grid-based decision"""
        
        # Always check: is straight path clear?
        straight_clear = self.is_straight_path_clear()
        
        if self.state == self.STATE_LANE_FOLLOWING:
            if straight_clear:
                # Path is clear -> continue lane following
                self.handle_lane_following()
            else:
                # Path blocked -> need to plan detour
                self.get_logger().warn('[GRID] Straight path blocked! Planning detour...')
                self.plan_detour()
        
        elif self.state == self.STATE_PATH_FOLLOWING:
            self.handle_path_following()
        
        elif self.state == self.STATE_RETURNING:
            self.handle_returning()

    def handle_lane_following(self):
        """Normal lane following - grid is clear ahead"""
        avoid_active = Bool()
        avoid_active.data = False
        self.pub_avoid_active.publish(avoid_active)
        
        max_vel = Float64()
        max_vel.data = self.lane_following_speed
        self.pub_max_vel.publish(max_vel)

    def plan_detour(self):
        """Plan path around obstacle"""
        # Start: slightly ahead of robot
        start_grid = self.world_to_grid(0.1, 0.0)
        
        # Goal: far end of grid, centered
        goal_x = self.grid_length - 0.2
        goal_y = 0.0
        goal_grid = self.world_to_grid(goal_x, goal_y)
        
        if start_grid is None or goal_grid is None:
            self.get_logger().error('[PLAN] Invalid start/goal!')
            return
        
        # Run A*
        path = self.find_path_astar(start_grid, goal_grid)
        
        if len(path) > 0:
            self.current_path = path
            self.path_index = 0
            self.state = self.STATE_PATH_FOLLOWING
            self.get_logger().info(f'[PLAN] Path found: {len(path)} waypoints')
        else:
            # No path -> STOP (forbidden zones blocking?)
            self.get_logger().error('[PLAN] NO PATH FOUND! Stopping.')
            self.pub_avoid_cmd.publish(Twist())  # Stop
            avoid_active = Bool()
            avoid_active.data = True
            self.pub_avoid_active.publish(avoid_active)

    def handle_path_following(self):
        """Follow planned path"""
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        if self.path_index >= len(self.current_path):
            # Path completed
            self.state = self.STATE_RETURNING
            self.get_logger().info('[PATH] Completed! Returning...')
            return
        
        # Get target waypoint
        target_x, target_y = self.current_path[self.path_index]
        
        # Check if reached
        dist = np.sqrt(target_x**2 + target_y**2)
        if dist < self.waypoint_threshold:
            self.path_index += 1
            if self.path_index < len(self.current_path):
                self.get_logger().info(f'[PATH] Waypoint {self.path_index}/{len(self.current_path)}', 
                                      throttle_duration_sec=0.5)
            return
        
        # Pure pursuit control
        twist = Twist()
        twist.linear.x = self.path_following_speed
        angle = np.arctan2(target_y, target_x)
        twist.angular.z = np.clip(self.steering_gain * angle, -1.2, 1.2)
        self.pub_avoid_cmd.publish(twist)
        
        # Re-plan if straight path becomes clear
        if self.is_straight_path_clear():
            self.state = self.STATE_RETURNING
            self.get_logger().info('[PATH] Straight clear again! Returning...')

    def handle_returning(self):
        """Return to centered lane following"""
        
        # Check if straight path is clear
        if not self.is_straight_path_clear():
            # Still blocked -> re-plan
            self.get_logger().warn('[RETURN] Path blocked again! Re-planning...')
            self.plan_detour()
            return
        
        # Check if centered
        if self.lane_state == 2:
            left_right_diff = abs(self.left_distance - self.right_distance)
            if left_right_diff < 100.0:
                # Well centered -> back to normal
                self.state = self.STATE_LANE_FOLLOWING
                self.get_logger().info('[RETURN] Centered! Back to lane following.')
                return
        
        # Steer to center
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        twist = Twist()
        twist.linear.x = self.path_following_speed * 0.7
        
        if self.left_distance < self.right_distance:
            twist.angular.z = -0.3
        else:
            twist.angular.z = 0.3
        
        self.pub_avoid_cmd.publish(twist)


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
