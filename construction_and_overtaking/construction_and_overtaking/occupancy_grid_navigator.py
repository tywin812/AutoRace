#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool, Header
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
import numpy as np


class OccupancyGridNavigator(Node):
    """Ultra-Pure Occupancy Grid Navigation with Passage Width Validation
    
    Grid -> Validate Width -> Plan -> Follow
    """

    def __init__(self):
        super().__init__('occupancy_grid_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_distance = self.create_subscription(Float64, '/lane_left_distance', self.left_distance_callback, 10)
        self.sub_right_distance = self.create_subscription(Float64, '/lane_right_distance', self.right_distance_callback, 10)
        self.sub_lane_state = self.create_subscription(UInt8, '/lane_detection_state', self.lane_state_callback, 10)
        
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
        
        # Grid parameters
        self.grid_resolution = 0.02  # 2cm cells for high precision
        self.grid_width = 2.0
        self.grid_length = 2.0
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Robot parameters
        self.robot_width = 0.10  # 10cm physical width
        self.robot_radius = 0.15  # 15cm with safety margin
        self.inflation_cells = int(np.ceil(self.robot_radius / self.grid_resolution))
        
        # Passage validation
        self.min_passage_width = 0.22  # 22cm minimum (robot 10cm + 12cm clearance)
        
        # Lane parameters
        self.lane_width_pixels = 450.0
        self.lane_width_meters = 0.6
        self.pixels_per_meter = self.lane_width_pixels / self.lane_width_meters
        self.lane_boundary_margin = 0.08
        
        # White line validation parameters
        self.min_lane_width_px = 400.0  # Minimum expected lane width (53cm)
        self.right_distance_history = []  # History for stability check
        self.history_length = 10  # 1 second at 10Hz
        self.max_jump_threshold_px = 150.0  # Maximum allowed sudden change
        self.white_line_valid = True  # Flag for current white line validity
        
        # Grid costs
        self.forbidden_cost = 1000.0  # Lane boundaries
        self.obstacle_cost = 100.0    # Physical obstacles
        
        # Control parameters
        self.speed = 0.18
        self.steering_gain = 3.0
        self.look_ahead_distance = 0.4
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.log_status)

        self.get_logger().info('=== Grid Navigator with Width Validation ===' )
        self.get_logger().info(f'Min passage width: {self.min_passage_width}m ({int(self.min_passage_width/self.grid_resolution)} cells)')
        self.get_logger().info(f'White line validation: min_width={self.min_lane_width_px}px, max_jump={self.max_jump_threshold_px}px')
        self.get_logger().info(f'Publishing to: /avoid_control, /avoid_active')

    def pixels_to_meters(self, pixel_distance):
        return pixel_distance / self.pixels_per_meter

    def validate_white_line(self):
        """
        Validates white line detection using two methods:
        1. Width check: Yellow line priority - reject if road too narrow
        2. Stability check: Reject sudden jumps (likely cones)
        
        Returns:
            bool: True if white line is trustworthy
        """
        if self.right_distance >= 999.0:
            return True  # No white line detected, nothing to validate
        
        # Check 1: Width validation (Yellow line priority)
        if self.left_distance < 999.0:
            current_width_px = self.left_distance + self.right_distance
            
            if current_width_px < self.min_lane_width_px:
                self.get_logger().warn(
                    f'[VALIDATE] Road too narrow! W={current_width_px:.0f}px < {self.min_lane_width_px:.0f}px. '
                    f'White line likely a CONE - ignoring.',
                    throttle_duration_sec=1.0
                )
                return False
        
        # Check 2: Stability validation
        if len(self.right_distance_history) >= 5:
            stable_distance = np.median(self.right_distance_history)
            
            if stable_distance < 999.0:
                change = abs(self.right_distance - stable_distance)
                
                if change > self.max_jump_threshold_px:
                    self.get_logger().warn(
                        f'[VALIDATE] Sudden jump! {stable_distance:.0f} -> {self.right_distance:.0f}px '
                        f'(Δ={change:.0f}px). Likely CONE - ignoring.',
                        throttle_duration_sec=1.0
                    )
                    return False
        
        return True  # All checks passed

    def get_lane_boundaries_in_meters(self):
        """
        Calculate lane boundaries with smart white line validation.
        """
        left_boundary = 999.0
        right_boundary = -999.0
        
        # Yellow line (left) - always trusted
        if self.left_distance < 999.0:
            left_boundary = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
        
        # White line (right) - validated
        if self.right_distance < 999.0:
            if self.white_line_valid:
                right_boundary = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
            else:
                # White line rejected - ignore it
                right_boundary = -999.0
                self.get_logger().debug('[LANE] Using yellow line only (white rejected)', throttle_duration_sec=2.0)
            
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
        """
        Checks if passage is wide enough for robot at given position.
        
        Args:
            grid_pos: (row, col) position on grid
            min_width_meters: minimum width (uses self.min_passage_width if None)
        
        Returns:
            bool: True if passage is sufficiently wide
        """
        if min_width_meters is None:
            min_width_meters = self.min_passage_width
            
        row, col = grid_pos
        min_width_cells = int(min_width_meters / self.grid_resolution)
        
        # Check width along Y axis (perpendicular to forward direction)
        # Go left from current position
        left_extent = 0
        for c in range(col - 1, -1, -1):
            if self.occupancy_grid[row, c] >= self.forbidden_cost:
                break
            left_extent += 1
            if left_extent >= min_width_cells // 2:
                break
        
        # Go right from current position
        right_extent = 0
        for c in range(col + 1, self.grid_w):
            if self.occupancy_grid[row, c] >= self.forbidden_cost:
                break
            right_extent += 1
            if right_extent >= min_width_cells // 2:
                break
        
        total_width_cells = left_extent + 1 + right_extent  # +1 for current cell
        
        return total_width_cells >= min_width_cells

    def build_occupancy_grid(self, lidar_points, stamp, frame_id):
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Get validated boundaries
        left_boundary, right_boundary = self.get_lane_boundaries_in_meters()
        
        # Draw lanes from paths (Curved)
        for path_points in [self.left_lane_path, self.right_lane_path]:
            for point in path_points:
                x, y = point
                grid_pos = self.world_to_grid(x, y)
                if grid_pos is not None:
                    row, col = grid_pos
                    self.occupancy_grid[row, col] = self.forbidden_cost
                    
                    # Thicken the line
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            r, c = row + dr, col + dc
                            if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                                self.occupancy_grid[r, c] = self.forbidden_cost

        # Fallback: Draw straight lines if paths are empty but we have distances
        if not self.left_lane_path and left_boundary < 999.0:
             for row in range(self.grid_h):
                x, y_dummy = self.grid_to_world(row, 0)
                if x > 1.0: continue
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    if y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost

        if not self.right_lane_path and right_boundary > -999.0:
             for row in range(self.grid_h):
                x, y_dummy = self.grid_to_world(row, 0)
                if x > 1.0: continue
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    if y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
        
        # Draw obstacles with inflation
        for point in lidar_points:
            x, y = point[0], point[1]
            grid_pos = self.world_to_grid(x, y)
            if grid_pos is not None:
                row, col = grid_pos
                
                # Inflate obstacles
                for dr in range(-self.inflation_cells, self.inflation_cells + 1):
                    for dc in range(-self.inflation_cells, self.inflation_cells + 1):
                        if dr*dr + dc*dc <= self.inflation_cells*self.inflation_cells:
                            r, c = row + dr, col + dc
                            if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                                # Don't overwrite forbidden zones (lanes) with lower cost
                                if self.occupancy_grid[r, c] < self.forbidden_cost:
                                    self.occupancy_grid[r, c] = max(
                                        self.occupancy_grid[r, c],
                                        self.obstacle_cost
                                    )
        
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
        """A* pathfinding with passage width validation."""
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        # Check goal validity
        if self.occupancy_grid[goal_row, goal_col] >= self.forbidden_cost:
            return []
        
        # Check if goal is in wide enough passage
        if not self.is_passage_wide_enough(goal_pos):
            self.get_logger().warn(
                f'[A*] Goal at ({goal_row},{goal_col}) is in narrow passage!',
                throttle_duration_sec=1.0
            )
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
                    
                    # Block by cost
                    if cell_cost >= self.forbidden_cost:
                        continue
                    
                    # Check passage width before adding to open set
                    if not self.is_passage_wide_enough(neighbor):
                        continue  # Skip narrow passages
                    
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
        """Right lane distance with stability tracking."""
        new_distance = 999.0 if msg.data < 0 else msg.data
        
        # Add to history
        self.right_distance_history.append(new_distance)
        if len(self.right_distance_history) > self.history_length:
            self.right_distance_history.pop(0)
        
        # Update raw distance
        self.right_distance = new_distance
        
        # Validate white line
        self.white_line_valid = self.validate_white_line()

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def log_status(self):
        left_b, right_b = self.get_lane_boundaries_in_meters()
        path_len = len(self.current_path)
        
        # Show validation status
        white_status = "✓" if self.white_line_valid else "✗"
        width_px = self.left_distance + self.right_distance if (self.left_distance < 999.0 and self.right_distance < 999.0) else 0
        
        target_info = ""
        if path_len > 0:
            target = self.find_lookahead_point()
            if target:
                tx, ty = target
                dist = np.sqrt(tx**2 + ty**2)
                angle = np.arctan2(ty, tx) * 180 / np.pi
                target_info = f"-> ({tx:.2f},{ty:.2f}) {dist:.2f}m {angle:.0f}°"
        
        self.get_logger().info(
            f'L={left_b:.2f} R={right_b:.2f} {white_status} W={width_px:.0f}px | Path={path_len} {target_info}'
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
        """Check for obstacles in forward corridor."""
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
        if not self.check_obstacles():
            avoid_active = Bool()
            avoid_active.data = False
            self.pub_avoid_active.publish(avoid_active)
            
            max_vel = Float64()
            max_vel.data = 0.22 
            self.pub_max_vel.publish(max_vel)
            
            self.current_path = []
            if hasattr(self, 'last_frame_id'):
                self.publish_debug_path([], self.last_frame_id)
            return

        # Dynamic Goal Selection with width validation
        goal_grid = None
        min_width_cells = int(self.min_passage_width / self.grid_resolution)
        
        start_scan_row = int(1.5 / self.grid_resolution)
        end_scan_row = int(0.5 / self.grid_resolution)
        
        for row in range(start_scan_row, end_scan_row, -1):
            if row >= self.grid_h:
                continue
            
            # Get free cells in this row
            free_cols = []
            for col in range(self.grid_w):
                if self.occupancy_grid[row, col] < self.obstacle_cost:
                    free_cols.append(col)
            
            # Require minimum width
            if len(free_cols) < min_width_cells:
                continue
            
            # Find contiguous segments
            segments = []
            if not free_cols:
                continue
                
            current_segment = [free_cols[0]]
            for i in range(1, len(free_cols)):
                if free_cols[i] == free_cols[i-1] + 1:
                    current_segment.append(free_cols[i])
                else:
                    # Only save wide segments
                    if len(current_segment) >= min_width_cells:
                        segments.append(current_segment)
                    current_segment = [free_cols[i]]
            
            # Check last segment
            if len(current_segment) >= min_width_cells:
                segments.append(current_segment)
            
            if not segments:
                continue
            
            # Pick the widest segment
            best_segment = max(segments, key=len)
            goal_col = best_segment[len(best_segment)//2]
            goal_grid = (row, goal_col)
            
            # Double-check with validation function
            if self.is_passage_wide_enough(goal_grid):
                self.get_logger().debug(
                    f'[GOAL] Found at row {row}, width={len(best_segment)} cells '
                    f'({len(best_segment)*self.grid_resolution:.2f}m)',
                    throttle_duration_sec=1.0
                )
                break
            else:
                goal_grid = None  # Reset and try next row
        
        start_grid = self.world_to_grid(0.05, 0.0)
        
        if goal_grid is None:
            self.get_logger().warn('[NO WIDE PASSAGE] All passages too narrow!', throttle_duration_sec=1.0)
            goal_grid = self.world_to_grid(0.5, 0.0)

        if start_grid is None or goal_grid is None:
            self.stop()
            return
        
        path = self.find_path_astar(start_grid, goal_grid)
        
        if len(path) == 0:
            self.get_logger().warn('[NO PATH] A* failed - narrow gaps detected', throttle_duration_sec=1.0)
            self.stop()
            return
        
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
