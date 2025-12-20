#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool
from geometry_msgs.msg import Twist
import numpy as np
from collections import deque


class OccupancyGridNavigator(Node):
    """Navigation using Occupancy Grid + A* Path Planning
    
    Key features:
    - Builds occupancy grid from LiDAR data
    - Marks cells behind lane lines as forbidden
    - Uses A* pathfinding to find safe path
    - Robust against confusing cones with lines
    """
    
    STATE_NORMAL = 0
    STATE_PLANNING = 1
    STATE_FOLLOWING_PATH = 2
    STATE_RETURNING = 3

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
        self.state = self.STATE_NORMAL
        self.left_distance = 999.0  # pixels
        self.right_distance = 999.0  # pixels
        self.lane_state = 0
        self.current_path = []  # List of (x, y) waypoints in meters
        self.path_index = 0
        
        # Grid parameters
        self.grid_resolution = 0.05  # meters per cell (5cm)
        self.grid_width = 4.0  # meters (2m left + 2m right)
        self.grid_length = 3.0  # meters (look ahead)
        self.grid_w = int(self.grid_width / self.grid_resolution)  # cells
        self.grid_h = int(self.grid_length / self.grid_resolution)  # cells
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Lane parameters
        self.image_center_x = 500.0
        self.lane_width_pixels = 600.0
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.05  # meters
        
        # Detection thresholds
        self.obstacle_threshold = 0.5  # meters - trigger planning
        self.clear_threshold = 1.5  # meters - path is clear
        self.occupied_threshold = 0.7  # grid cell probability to mark as occupied
        self.forbidden_cost = 100.0  # cost for forbidden cells (behind lines)
        self.obstacle_cost = 10.0  # cost for obstacle cells
        
        # Control parameters
        self.avoidance_speed = 0.15
        self.normal_speed = 0.22
        self.look_ahead_distance = 0.3  # meters for path following
        self.path_following_threshold = 0.1  # meters - waypoint reached
        self.waypoint_angular_gain = 2.0  # proportional gain for steering
        
        # Timers
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.log_status)

        self.get_logger().info('Occupancy Grid Navigator initialized')
        self.get_logger().info(f'Grid: {self.grid_w}x{self.grid_h} cells ({self.grid_resolution}m resolution)')
        self.get_logger().info(f'Lane filtering: {self.pixels_per_meter:.1f} px/m')

    def pixels_to_meters(self, pixel_distance):
        """Convert pixel distance from camera to meters"""
        return pixel_distance / self.pixels_per_meter

    def get_lane_boundaries_in_meters(self):
        """Get left and right lane boundaries in robot frame (meters)
        Returns: (left_boundary_y, right_boundary_y)
        Left boundary is positive Y, right boundary is negative Y
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
        """Convert world coordinates (meters, robot frame) to grid indices
        Robot is at bottom center of grid
        X: forward (0 to grid_length)
        Y: lateral (-grid_width/2 to +grid_width/2)
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
        """Convert grid indices to world coordinates (meters, robot frame)
        Returns: (x, y)
        """
        x = (row + 0.5) * self.grid_resolution
        y = (col + 0.5) * self.grid_resolution - self.grid_width/2
        return (x, y)

    def build_occupancy_grid(self, lidar_points):
        """Build occupancy grid from LiDAR points and lane boundaries
        
        Args:
            lidar_points: Nx2 array of (x, y) points in meters
        """
        # Reset grid
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Mark obstacles from LiDAR
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                # Inflate obstacle slightly (3x3 kernel)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        r, c = row + dr, col + dc
                        if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                            self.occupancy_grid[r, c] = self.obstacle_cost
        
        # Mark forbidden zones (behind lane lines)
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        
        if self.lane_state > 0:  # Lane lines detected
            for row in range(self.grid_h):
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    
                    # Mark cells beyond left line as forbidden
                    if left_boundary < 999.0 and y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
                    
                    # Mark cells beyond right line as forbidden
                    if right_boundary > -999.0 and y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost

    def find_path_astar(self, start_pos, goal_pos):
        """A* pathfinding on occupancy grid
        
        Args:
            start_pos: (row, col) starting cell
            goal_pos: (row, col) goal cell
            
        Returns:
            List of (x, y) waypoints in world coordinates, or empty list if no path
        """
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        # Priority queue: (f_score, counter, (row, col))
        counter = 0
        open_set = [(0, counter, start_pos)]
        counter += 1
        
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: self.heuristic(start_pos, goal_pos)}
        
        visited = set()
        
        while open_set:
            # Get node with lowest f_score
            open_set.sort(key=lambda x: x[0])
            current_f, _, current = open_set.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check if reached goal
            if current == goal_pos:
                return self.reconstruct_path(came_from, current)
            
            # Check neighbors (8-connected)
            row, col = current
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    neighbor_row = row + dr
                    neighbor_col = col + dc
                    
                    # Check bounds
                    if not (0 <= neighbor_row < self.grid_h and 0 <= neighbor_col < self.grid_w):
                        continue
                    
                    neighbor = (neighbor_row, neighbor_col)
                    
                    if neighbor in visited:
                        continue
                    
                    # Calculate movement cost
                    diagonal = (dr != 0 and dc != 0)
                    move_cost = 1.414 if diagonal else 1.0
                    
                    # Add grid cell cost
                    cell_cost = self.occupancy_grid[neighbor_row, neighbor_col]
                    tentative_g = g_score[current] + move_cost + cell_cost
                    
                    # Skip if forbidden
                    if cell_cost >= self.forbidden_cost:
                        continue
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, goal_pos)
                        f_score[neighbor] = f
                        open_set.append((f, counter, neighbor))
                        counter += 1
        
        # No path found
        return []

    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic for A*"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from A* came_from dict
        Returns list of (x, y) waypoints in world coordinates
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        # Convert grid coordinates to world coordinates
        world_path = []
        for row, col in path:
            x, y = self.grid_to_world(row, col)
            world_path.append((x, y))
        
        # Simplify path (keep every 3rd waypoint to reduce oscillations)
        simplified_path = world_path[::3]
        if len(world_path) > 0 and world_path[-1] not in simplified_path:
            simplified_path.append(world_path[-1])
        
        return simplified_path

    def get_closest_obstacle_distance(self):
        """Get minimum distance to any obstacle in front corridor"""
        min_dist = 999.0
        corridor_width = 0.4  # meters
        
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                x, y = self.grid_to_world(row, col)
                
                # Check if in front corridor
                if abs(y) < corridor_width / 2:
                    cost = self.occupancy_grid[row, col]
                    if cost >= self.obstacle_cost and cost < self.forbidden_cost:
                        dist = x
                        min_dist = min(min_dist, dist)
        
        return min_dist

    def lidar_callback(self, scan_msg):
        """Process LiDAR scan and build occupancy grid"""
        ranges = np.array(scan_msg.ranges)
        ranges[ranges == 0] = float('inf')
        ranges[ranges == float('-inf')] = float('inf')
        ranges[np.isnan(ranges)] = float('inf')

        # Rotate to have 0° at front
        ranges = np.roll(ranges, len(ranges)//2)
        angles = np.linspace(np.pi, 3*np.pi, len(ranges), endpoint=False)

        # Convert to Cartesian coordinates
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Filter: only points in grid area and valid
        mask = (x > 0) & (x < self.grid_length) & (np.abs(y) < self.grid_width/2) & (ranges < 10.0)
        
        lidar_points = np.column_stack([x[mask], y[mask]])
        
        # Build occupancy grid
        self.build_occupancy_grid(lidar_points)

    def left_distance_callback(self, msg):
        self.left_distance = 999.0 if msg.data < 0 else msg.data

    def right_distance_callback(self, msg):
        self.right_distance = 999.0 if msg.data < 0 else msg.data

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def log_status(self):
        """Log current status"""
        state_names = {
            0: "NORMAL",
            1: "PLANNING",
            2: "FOLLOWING_PATH",
            3: "RETURNING"
        }
        current_state_name = state_names.get(self.state, "UNKNOWN")
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        min_obstacle_dist = self.get_closest_obstacle_distance()
        
        self.get_logger().info(
            f'State: {current_state_name} | '
            f'Lanes: L={left_boundary:.2f}m R={right_boundary:.2f}m | '
            f'Obstacle: {min_obstacle_dist:.2f}m | '
            f'Path points: {len(self.current_path)}'
        )

    def control_loop(self):
        """Main control loop"""
        if self.state == self.STATE_NORMAL:
            self.handle_normal_state()
        elif self.state == self.STATE_PLANNING:
            self.handle_planning_state()
        elif self.state == self.STATE_FOLLOWING_PATH:
            self.handle_following_path_state()
        elif self.state == self.STATE_RETURNING:
            self.handle_returning_state()

    def handle_normal_state(self):
        """Normal driving - follow lane"""
        avoid_active = Bool()
        avoid_active.data = False
        self.pub_avoid_active.publish(avoid_active)

        # Adaptive speed based on obstacle distance
        min_obstacle_dist = self.get_closest_obstacle_distance()
        max_vel = Float64()
        
        if min_obstacle_dist < self.clear_threshold:
            # Slow down as approaching obstacle
            scale = (min_obstacle_dist - self.obstacle_threshold) / (self.clear_threshold - self.obstacle_threshold)
            scale = max(0.0, min(1.0, scale))
            max_vel.data = 0.05 + (scale ** 0.5) * (self.normal_speed - 0.05)
        else:
            max_vel.data = self.normal_speed
        
        self.pub_max_vel.publish(max_vel)

        # Trigger planning if obstacle detected
        if min_obstacle_dist < self.obstacle_threshold:
            self.state = self.STATE_PLANNING
            self.get_logger().warn(f'[ALERT] Obstacle at {min_obstacle_dist:.2f}m! Starting path planning...')

    def handle_planning_state(self):
        """Plan path around obstacle using A*"""
        # Start position: robot at bottom center of grid
        start_grid = self.world_to_grid(0.1, 0.0)  # slightly ahead
        
        # Goal position: straight ahead at far end of grid
        goal_x = self.grid_length - 0.2
        goal_y = 0.0
        goal_grid = self.world_to_grid(goal_x, goal_y)
        
        if start_grid is None or goal_grid is None:
            self.get_logger().error('[PLANNING] Invalid start/goal position!')
            self.state = self.STATE_NORMAL
            return
        
        # Find path
        path = self.find_path_astar(start_grid, goal_grid)
        
        if len(path) > 0:
            self.current_path = path
            self.path_index = 0
            self.state = self.STATE_FOLLOWING_PATH
            self.get_logger().info(f'[SUCCESS] Path found with {len(path)} waypoints!')
        else:
            # No path found - stop and wait
            self.get_logger().warn('[PLANNING] No valid path found! Stopping.')
            self.pub_avoid_cmd.publish(Twist())  # Stop
            avoid_active = Bool()
            avoid_active.data = True
            self.pub_avoid_active.publish(avoid_active)
            # Retry planning next cycle

    def handle_following_path_state(self):
        """Follow planned path using pure pursuit"""
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        if len(self.current_path) == 0 or self.path_index >= len(self.current_path):
            # Path completed
            self.state = self.STATE_RETURNING
            self.get_logger().info('[PATH] Path completed! Returning to lane...')
            return
        
        # Get current target waypoint
        target_x, target_y = self.current_path[self.path_index]
        
        # Check if waypoint reached
        distance_to_waypoint = np.sqrt(target_x**2 + target_y**2)
        if distance_to_waypoint < self.path_following_threshold:
            self.path_index += 1
            if self.path_index < len(self.current_path):
                target_x, target_y = self.current_path[self.path_index]
                self.get_logger().info(f'[PATH] Waypoint {self.path_index}/{len(self.current_path)} reached')
            else:
                self.state = self.STATE_RETURNING
                return
        
        # Pure pursuit control
        twist = Twist()
        twist.linear.x = self.avoidance_speed
        
        # Calculate angular velocity to steer towards waypoint
        angle_to_waypoint = np.arctan2(target_y, target_x)
        twist.angular.z = self.waypoint_angular_gain * angle_to_waypoint
        
        # Limit angular velocity
        max_angular = 1.0
        twist.angular.z = np.clip(twist.angular.z, -max_angular, max_angular)
        
        self.pub_avoid_cmd.publish(twist)
        
        # Check if new obstacle appeared
        min_obstacle_dist = self.get_closest_obstacle_distance()
        if min_obstacle_dist < self.obstacle_threshold * 0.5:
            self.get_logger().warn('[PATH] New obstacle detected! Re-planning...')
            self.state = self.STATE_PLANNING

    def handle_returning_state(self):
        """Return to lane following after obstacle avoidance"""
        # Check if obstacle still present
        min_obstacle_dist = self.get_closest_obstacle_distance()
        if min_obstacle_dist < self.obstacle_threshold:
            self.state = self.STATE_PLANNING
            return
        
        # Check if centered in lane
        if self.lane_state == 2:  # Both lines detected
            left_right_diff = abs(self.left_distance - self.right_distance)
            if left_right_diff < 50.0:  # Well centered
                self.state = self.STATE_NORMAL
                self.get_logger().info('[RETURN] Successfully returned to lane!')
                return
        
        # Gentle steering to center
        avoid_active = Bool()
        avoid_active.data = True
        self.pub_avoid_active.publish(avoid_active)
        
        twist = Twist()
        twist.linear.x = self.avoidance_speed * 0.8
        
        # Steer towards center based on lane distances
        if self.left_distance < self.right_distance:
            twist.angular.z = -0.2  # Turn right
        else:
            twist.angular.z = 0.2  # Turn left
        
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
