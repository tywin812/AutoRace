#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool, Header
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
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
        self.grid_resolution = 0.02  # Reduced to 2cm for finer grid
        self.grid_width = 2.0
        self.grid_length = 2.0
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Robot parameters
        self.robot_radius = 0.15  # Increased to 15cm for safety margin
        self.inflation_cells = int(np.ceil(self.robot_radius / self.grid_resolution))
        
        # Lane parameters
        self.lane_width_pixels = 450.0
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

    def build_occupancy_grid(self, lidar_points, stamp, frame_id):
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
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
        # This handles cases where path message hasn't arrived yet or is empty
        if not self.left_lane_path and self.left_distance < 999.0:
             left_boundary = self.pixels_to_meters(self.left_distance) + self.lane_boundary_margin
             for row in range(self.grid_h):
                x, y_dummy = self.grid_to_world(row, 0)
                if x > 1.0: continue
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    if y > left_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost

        if not self.right_lane_path and self.right_distance < 999.0:
             right_boundary = -(self.pixels_to_meters(self.right_distance) + self.lane_boundary_margin)
             for row in range(self.grid_h):
                x, y_dummy = self.grid_to_world(row, 0)
                if x > 1.0: continue
                for col in range(self.grid_w):
                    x, y = self.grid_to_world(row, col)
                    if y < right_boundary:
                        self.occupancy_grid[row, col] = self.forbidden_cost
        
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
        # Force zero time to ensure visualization works in RViz despite TF delays
        grid_msg.header.stamp = rclpy.time.Time().to_msg()
        grid_msg.header.frame_id = frame_id
        
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_h  # X-axis size
        grid_msg.info.height = self.grid_w # Y-axis size
        
        # Origin is at (0, -width/2) relative to base_scan
        grid_msg.info.origin.position.x = 0.0
        grid_msg.info.origin.position.y = -self.grid_width / 2.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        # Convert float grid to int8 [0-100]
        # 0 = free, 100 = occupied, -1 = unknown
        flat_grid = np.zeros(self.grid_h * self.grid_w, dtype=np.int8)
        
        # Transpose because ROS grid is row-major (y then x), but our grid is [row, col] where row is x
        # Actually ROS OccupancyGrid data is row-major, starting from (0,0).
        # Index = y * width + x.
        # Our grid: row is X (0..L), col is Y (-W/2..W/2).
        # So we need to map our [row, col] to ROS [y, x].
        # But wait, ROS map usually has X forward, Y left.
        # If we set origin correctly, we can just flatten it?
        # Let's check:
        # ROS data[i] corresponds to cell (i % width, i / width).
        # i % width is x-index (column), i / width is y-index (row).
        # So data is ordered by y, then x.
        # Our grid is [row, col] -> [x, y].
        # So we need to transpose or iterate correctly.
        
        # Let's just iterate and fill
        for r in range(self.grid_h): # x
            for c in range(self.grid_w): # y
                val = self.occupancy_grid[r, c]
                ros_val = 0
                if val >= self.forbidden_cost:
                    ros_val = 100
                elif val >= self.obstacle_cost:
                    ros_val = 50
                else:
                    ros_val = 0
                
                # ROS grid index: y * width + x
                # Our c is y-index (0..W), r is x-index (0..H)
                # But wait, ROS OccupancyGrid width is number of cells in x-direction?
                # No, width is number of cells in x-axis of the map image.
                # Usually map x is world x.
                # Let's assume standard ROS map convention:
                # width is size in X, height is size in Y?
                # No, usually width is columns, height is rows.
                # If we want X forward, Y left.
                # Let's set width = grid_h (X size), height = grid_w (Y size)?
                # No, that's confusing.
                
                # Let's stick to:
                # grid_msg.info.width = self.grid_h (cells along X axis)
                # grid_msg.info.height = self.grid_w (cells along Y axis)
                # And origin orientation?
                
                # Standard:
                # Data is (0,0), (1,0), ... (W-1, 0), (0, 1), ...
                # (x, y)
                
                # Our grid:
                # row 0..H (X axis)
                # col 0..W (Y axis)
                
                # If we map our X to ROS X, and our Y to ROS Y.
                # Then ROS Width = H (size in X), ROS Height = W (size in Y).
                # Index = y * ROS_Width + x
                # Index = c * H + r
                
                idx = c * self.grid_h + r
                # Wait, this implies width is H.
                
        # Let's try to just publish it as is and adjust in Rviz if needed.
        # But better to be correct.
        
        # Let's define ROS Map X axis as Robot X axis.
        # ROS Map Y axis as Robot Y axis.
        # So ROS Width (cells in X) = self.grid_h
        # ROS Height (cells in Y) = self.grid_w
        
        grid_msg.info.width = self.grid_h
        grid_msg.info.height = self.grid_w
        
        # Flattening:
        # We need data ordered by Y then X.
        # i.e. for y=0: x=0..H-1
        #      for y=1: x=0..H-1
        
        # Our grid is [x, y].
        # We want [y, x] order for flattening?
        # data = [ (0,0), (1,0), (2,0)... (0,1), (1,1)... ]
        # Yes.
        
        # So we can transpose our grid to [y, x] then flatten.
        grid_t = self.occupancy_grid.T # Now [col, row] -> [y, x]
        
        # Map values
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
        # Filter invalid ranges
        ranges[ranges == 0] = float('inf')
        ranges[ranges < scan_msg.range_min] = float('inf')
        ranges[ranges > scan_msg.range_max] = float('inf')
        ranges[np.isnan(ranges)] = float('inf')

        # Calculate angles using actual scan parameters
        # This is more accurate than linspace as it respects the sensor's actual calibration
        angles = np.arange(
            scan_msg.angle_min, 
            scan_msg.angle_min + len(ranges) * scan_msg.angle_increment, 
            scan_msg.angle_increment
        )
        
        # Safety check for array sizes
        if len(angles) > len(ranges):
            angles = angles[:len(ranges)]
        elif len(angles) < len(ranges):
             angles = np.pad(angles, (0, len(ranges) - len(angles)), 'edge')

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Transform to base_link (Rotate 180 deg around Z)
        # The lidar is mounted rotated 180 degrees relative to base_link
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

    def check_obstacles(self):
        # Define ROI (0 to 1.5m ahead, +/- 0.3m width)
        min_x, max_x = 0.0, 1.5
        min_y, max_y = -0.3, 0.3
        
        start_row = int(min_x / self.grid_resolution)
        end_row = int(max_x / self.grid_resolution)
        
        start_col = int((min_y + self.grid_width/2) / self.grid_resolution)
        end_col = int((max_y + self.grid_width/2) / self.grid_resolution)
        
        # Clamp
        start_row = max(0, start_row)
        end_row = min(self.grid_h, end_row)
        start_col = max(0, start_col)
        end_col = min(self.grid_w, end_col)
        
        # Check for obstacles (cost == 100)
        if start_row >= end_row or start_col >= end_col:
            return False
            
        roi = self.occupancy_grid[start_row:end_row, start_col:end_col]
        return np.any(roi == self.obstacle_cost)

    def control_loop(self):
        # Check for obstacles first
        if not self.check_obstacles():
            # No obstacles, let lane follower drive
            avoid_active = Bool()
            avoid_active.data = False
            self.pub_avoid_active.publish(avoid_active)
            
            # Publish max vel (normal speed)
            max_vel = Float64()
            max_vel.data = 0.22 
            self.pub_max_vel.publish(max_vel)
            
            # Clear path debug and state
            self.current_path = []
            if hasattr(self, 'last_frame_id'):
                self.publish_debug_path([], self.last_frame_id)
            return

        # Dynamic Goal Selection
        # Instead of a fixed point, find the furthest reachable free space
        # Scan rows from far to near
        goal_grid = None
        
        # Start scanning from 1.5m down to 0.5m
        start_scan_row = int(1.5 / self.grid_resolution)
        end_scan_row = int(0.5 / self.grid_resolution)
        
        for row in range(start_scan_row, end_scan_row, -1):
            if row >= self.grid_h: continue
            
            # Get free cells in this row
            free_cols = []
            for col in range(self.grid_w):
                if self.occupancy_grid[row, col] < self.obstacle_cost:
                    free_cols.append(col)
            
            if len(free_cols) > 5: # Need at least 10cm width
                # Find the largest contiguous segment
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
                
                # Pick the largest segment
                best_segment = max(segments, key=len)
                
                # Goal is the middle of the largest segment
                goal_col = best_segment[len(best_segment)//2]
                goal_grid = (row, goal_col)
                break
        
        start_grid = self.world_to_grid(0.05, 0.0)
        
        # Fallback if no goal found (e.g. blocked)
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
        # We need frame_id here, but control_loop doesn't have it.
        # We can store the last frame_id in lidar_callback
        if hasattr(self, 'last_frame_id'):
            # Use the last known timestamp if possible, otherwise current time
            # But publish_debug_path uses get_clock().now() currently.
            # Let's just pass frame_id.
            self.publish_debug_path(path, self.last_frame_id)
        self.follow_path()

    def publish_debug_path(self, path, frame_id):
        path_msg = Path()
        path_msg.header = Header()
        # Force zero time to ensure visualization works in RViz despite TF delays
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
