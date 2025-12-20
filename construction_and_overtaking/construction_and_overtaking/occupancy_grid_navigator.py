#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, UInt8, Bool, Header
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
import numpy as np


class ObstacleClassifier:
    """Classifies obstacles by danger level based on trajectory."""
    
    def __init__(self, robot_radius=0.12):
        self.robot_radius = robot_radius
    
    def classify_obstacle(self, obstacle_pos, trajectory):
        """
        Args:
            obstacle_pos: (x, y) in robot frame
            trajectory: List[(x, y, width)] predicted path
        
        Returns:
            str: "CRITICAL", "WARNING", "LATERAL", "SAFE"
        """
        if len(trajectory) == 0:
            return "SAFE"
        
        obs_x, obs_y = obstacle_pos
        
        # Check distance to trajectory
        min_dist_to_trajectory = float('inf')
        trajectory_width_at_obstacle = 0.5
        closest_traj_x = 0
        
        for traj_x, traj_y, traj_width in trajectory:
            dist = np.sqrt((obs_x - traj_x)**2 + (obs_y - traj_y)**2)
            
            if dist < min_dist_to_trajectory:
                min_dist_to_trajectory = dist
                trajectory_width_at_obstacle = traj_width
                closest_traj_x = traj_x
        
        # Safety corridor
        safety_corridor = (trajectory_width_at_obstacle / 2.0) + self.robot_radius
        
        # Classification
        if min_dist_to_trajectory < safety_corridor * 0.7:
            if obs_x < 0.5:
                return "CRITICAL"
            else:
                return "WARNING"
        elif min_dist_to_trajectory < safety_corridor * 1.2:
            return "WARNING"
        elif obs_x > 0 and abs(obs_y) > safety_corridor:
            return "LATERAL"
        else:
            return "SAFE"


class IntelligentModeController:
    """Intelligent mode switching with hysteresis."""
    
    def __init__(self):
        self.current_mode = "LANE_FOLLOWING"
        self.critical_obstacle_time = 0.0
        self.warning_obstacle_time = 0.0
    
    def decide_mode(self, obstacle_counts, lane_quality, dt):
        """
        Args:
            obstacle_counts: dict {"CRITICAL": int, "WARNING": int, ...}
            lane_quality: float 0-1
            dt: float time since last call
        
        Returns:
            str: "LANE_FOLLOWING" or "OBSTACLE_AVOIDANCE"
        """
        critical_count = obstacle_counts.get("CRITICAL", 0)
        warning_count = obstacle_counts.get("WARNING", 0)
        
        # Update timers with decay
        if critical_count > 0:
            self.critical_obstacle_time += dt
        else:
            self.critical_obstacle_time = max(0, self.critical_obstacle_time - dt * 2)
        
        if warning_count > 0:
            self.warning_obstacle_time += dt
        else:
            self.warning_obstacle_time = max(0, self.warning_obstacle_time - dt * 2)
        
        # Decision rules
        
        # Rule 1: Critical obstacle confirmed
        if self.critical_obstacle_time > 0.2:
            return "OBSTACLE_AVOIDANCE"
        
        # Rule 2: Multiple warnings with poor lanes
        if warning_count >= 2 and lane_quality < 0.5:
            return "OBSTACLE_AVOIDANCE"
        
        # Rule 3: Single warning with good lanes (likely turn cone)
        if warning_count == 1 and lane_quality > 0.7:
            return "LANE_FOLLOWING"
        
        # Rule 4: Only lateral obstacles
        if obstacle_counts.get("LATERAL", 0) > 0 and critical_count == 0 and warning_count == 0:
            return "LANE_FOLLOWING"
        
        # Rule 5: Clear path with good lanes
        if lane_quality > 0.6 and critical_count == 0 and warning_count == 0:
            return "LANE_FOLLOWING"
        
        # Rule 6: Poor lanes without obstacles - use grid navigation
        if lane_quality < 0.3 and critical_count == 0:
            return "OBSTACLE_AVOIDANCE"
        
        # Default: maintain current mode
        return self.current_mode


class OccupancyGridNavigator(Node):
    """Intelligent trajectory-based obstacle avoidance."""

    def __init__(self):
        super().__init__('occupancy_grid_navigator')

        # Subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_left_path = self.create_subscription(Path, '/detect/lane_left_path', self.left_path_callback, 10)
        self.sub_right_path = self.create_subscription(Path, '/detect/lane_right_path', self.right_path_callback, 10)
        self.sub_lane_state = self.create_subscription(UInt8, '/lane_detection_state', self.lane_state_callback, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, '/avoid_control', 10)
        self.pub_avoid_active = self.create_publisher(Bool, '/avoid_active', 10)
        self.pub_max_vel = self.create_publisher(Float64, '/control/max_vel', 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/debug/grid', 10)
        self.pub_path = self.create_publisher(Path, '/debug/path', 10)
        self.pub_trajectory = self.create_publisher(Path, '/debug/predicted_trajectory', 10)

        # State
        self.lane_state = 0
        self.current_path = []
        self.left_lane_path = []
        self.right_lane_path = []
        self.last_left_path_time = self.get_clock().now()
        self.last_right_path_time = self.get_clock().now()
        self.last_lidar_points = []
        
        # Grid parameters
        self.grid_resolution = 0.02
        self.grid_width = 2.0
        self.grid_min_x = -0.5
        self.grid_max_x = 1.5
        self.grid_length = self.grid_max_x - self.grid_min_x
        self.grid_w = int(self.grid_width / self.grid_resolution)
        self.grid_h = int(self.grid_length / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Robot parameters
        self.robot_radius = 0.12
        self.inflation_cells = int(np.ceil(self.robot_radius / self.grid_resolution))
        
        # Grid costs
        self.forbidden_cost = 100.0
        self.obstacle_cost = 100.0
        
        # Control parameters
        self.speed = 0.08
        self.steering_gain = 1.5
        self.look_ahead_distance = 0.25
        
        # Intelligence components
        self.obstacle_classifier = ObstacleClassifier(self.robot_radius)
        self.mode_controller = IntelligentModeController()
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        self.last_control_time = self.get_clock().now()

        self.get_logger().info('=== Intelligent Grid Navigator ===' )
        self.get_logger().info('Trajectory-based obstacle avoidance active')

    def predict_lane_trajectory(self, horizon_distance=1.5):
        """
        Predicts robot trajectory if following lanes.
        
        Returns:
            List[(x, y, width)] - trajectory points with corridor width
        """
        trajectory = []
        
        if len(self.left_lane_path) > 0 and len(self.right_lane_path) > 0:
            # Both lanes visible - trajectory is center
            min_len = min(len(self.left_lane_path), len(self.right_lane_path))
            
            for i in range(min_len):
                left_pt = self.left_lane_path[i]
                right_pt = self.right_lane_path[i]
                
                center_x = (left_pt[0] + right_pt[0]) / 2.0
                center_y = (left_pt[1] + right_pt[1]) / 2.0
                width = np.sqrt((right_pt[0] - left_pt[0])**2 + (right_pt[1] - left_pt[1])**2)
                
                if center_x <= horizon_distance:
                    trajectory.append((center_x, center_y, width))
        
        elif len(self.left_lane_path) > 0:
            # Only left lane - offset to right
            for pt in self.left_lane_path:
                if pt[0] <= horizon_distance:
                    trajectory.append((pt[0], pt[1] - 0.3, 0.6))
        
        elif len(self.right_lane_path) > 0:
            # Only right lane - offset to left
            for pt in self.right_lane_path:
                if pt[0] <= horizon_distance:
                    trajectory.append((pt[0], pt[1] + 0.3, 0.6))
        
        else:
            # No lanes - straight ahead
            for x in np.arange(0, horizon_distance, 0.1):
                trajectory.append((x, 0.0, 0.5))
        
        return trajectory

    def angular_obstacle_filter(self, lidar_points, trajectory):
        """
        Filters obstacles by angle relative to trajectory direction.
        """
        if len(trajectory) < 3:
            return lidar_points
        
        # Calculate trajectory curvature
        near_pt = None
        far_pt = None
        
        for pt in trajectory:
            if near_pt is None and pt[0] > 0.3:
                near_pt = pt
            if far_pt is None and pt[0] > 0.6:
                far_pt = pt
            if near_pt and far_pt:
                break
        
        if near_pt is None or far_pt is None:
            return lidar_points
        
        # Turn direction (positive = left, negative = right)
        turn_direction = far_pt[1] - near_pt[1]
        
        filtered = []
        for x, y in lidar_points:
            angle = np.arctan2(y, x)
            
            # Turning left - ignore far right obstacles
            if turn_direction > 0.05:
                if angle < -0.7 and x > 0.3:
                    continue
            
            # Turning right - ignore far left obstacles
            elif turn_direction < -0.05:
                if angle > 0.7 and x > 0.3:
                    continue
            
            filtered.append((x, y))
        
        return filtered

    def calculate_lane_quality(self):
        """
        Evaluates lane detection quality (0-1).
        """
        current_time = self.get_clock().now()
        
        # Age of lane data
        left_age = (current_time - self.last_left_path_time).nanoseconds / 1e9
        right_age = (current_time - self.last_right_path_time).nanoseconds / 1e9
        
        left_freshness = max(0, 1.0 - left_age / 1.0)
        right_freshness = max(0, 1.0 - right_age / 1.0)
        
        # Density of points
        left_density = min(1.0, len(self.left_lane_path) / 30.0)
        right_density = min(1.0, len(self.right_lane_path) / 30.0)
        
        # Combined quality
        if len(self.left_lane_path) > 5 and len(self.right_lane_path) > 5:
            quality = 0.5 * (left_freshness + right_freshness) + \
                     0.3 * (left_density + right_density) / 2.0 + 0.2
        elif len(self.left_lane_path) > 5:
            quality = 0.6 * left_freshness + 0.3 * left_density
        elif len(self.right_lane_path) > 5:
            quality = 0.6 * right_freshness + 0.3 * right_density
        else:
            quality = 0.0
        
        return np.clip(quality, 0.0, 1.0)

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
        
        # Draw lane paths (always, no timeout)
        paths_to_draw = []
        if len(self.left_lane_path) > 0:
            paths_to_draw.append(self.left_lane_path)
        if len(self.right_lane_path) > 0:
            paths_to_draw.append(self.right_lane_path)

        for path_points in paths_to_draw:
            if len(path_points) < 2:
                continue
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if dist < 0.15:
                    self.draw_thick_line_on_grid(p1, p2, thickness_cells=5)
        
        # Draw obstacles
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

    def publish_trajectory(self, trajectory, frame_id):
        """Publish predicted trajectory for debugging."""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        
        for x, y, width in trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.pub_trajectory.publish(path_msg)

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
        
        x_robot = -x
        y_robot = -y
        
        # Wide FOV for turn detection
        angles_robot = np.arctan2(y_robot, x_robot)
        fov_limit = 120 * np.pi / 180
        mask_fov = np.abs(angles_robot) < fov_limit
        
        mask = (x_robot > self.grid_min_x) & (x_robot < self.grid_max_x) & \
               (np.abs(y_robot) < self.grid_width/2) & (ranges < 10.0) & mask_fov
        
        self.last_lidar_points = np.column_stack([x_robot[mask], y_robot[mask]]).tolist()
        self.last_frame_id = "robot/base_link"
        self.last_scan_time = scan_msg.header.stamp

    def left_path_callback(self, msg):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.left_lane_path = points
        self.last_left_path_time = self.get_clock().now()

    def right_path_callback(self, msg):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.right_lane_path = points
        self.last_right_path_time = self.get_clock().now()

    def lane_state_callback(self, msg):
        self.lane_state = msg.data

    def control_loop(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time
        
        # 1. Predict lane trajectory
        trajectory = self.predict_lane_trajectory(horizon_distance=1.5)
        
        # Publish for debugging
        if len(trajectory) > 0:
            self.publish_trajectory(trajectory, "robot/base_link")
        
        # 2. Filter lidar by angle
        filtered_lidar = self.angular_obstacle_filter(self.last_lidar_points, trajectory)
        
        # 3. Build occupancy grid
        if hasattr(self, 'last_scan_time'):
            self.build_occupancy_grid(filtered_lidar, self.last_scan_time, self.last_frame_id)
        
        # 4. Classify obstacles
        obstacle_counts = {"CRITICAL": 0, "WARNING": 0, "LATERAL": 0, "SAFE": 0}
        
        for point in filtered_lidar:
            classification = self.obstacle_classifier.classify_obstacle(point, trajectory)
            obstacle_counts[classification] += 1
        
        # 5. Evaluate lane quality
        lane_quality = self.calculate_lane_quality()
        
        # 6. Decide mode
        mode = self.mode_controller.decide_mode(obstacle_counts, lane_quality, dt)
        self.mode_controller.current_mode = mode
        
        # 7. Execute mode
        if mode == "LANE_FOLLOWING":
            avoid_active = Bool()
            avoid_active.data = False
            self.pub_avoid_active.publish(avoid_active)
            
            self.get_logger().info(
                f'[LANE] Q={lane_quality:.2f} | '
                f'C={obstacle_counts["CRITICAL"]} W={obstacle_counts["WARNING"]} '
                f'L={obstacle_counts["LATERAL"]} S={obstacle_counts["SAFE"]}',
                throttle_duration_sec=0.5
            )
        
        else:  # OBSTACLE_AVOIDANCE
            path = self.plan_avoidance_path()
            
            if len(path) > 0:
                self.current_path = path
                self.follow_path()
                
                self.get_logger().info(
                    f'[AVOID] Path={len(path)} | '
                    f'C={obstacle_counts["CRITICAL"]} W={obstacle_counts["WARNING"]}',
                    throttle_duration_sec=0.5
                )
            else:
                self.stop()

    def plan_avoidance_path(self):
        """Plan path using A* on occupancy grid."""
        start_grid = self.world_to_grid(0.05, 0.0)
        
        # Dynamic goal selection
        goal_grid = None
        start_scan_row = int((1.2 - self.grid_min_x) / self.grid_resolution)
        end_scan_row = int((0.4 - self.grid_min_x) / self.grid_resolution)
        
        for row in range(start_scan_row, end_scan_row, -1):
            if row >= self.grid_h or row < 0:
                continue
            
            free_cols = [col for col in range(self.grid_w) 
                        if self.occupancy_grid[row, col] < self.obstacle_cost]
            
            if len(free_cols) > 8:
                segments = []
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
        
        if goal_grid is None:
            goal_grid = self.world_to_grid(0.5, 0.0)
        
        if start_grid is None or goal_grid is None:
            return []
        
        return self.find_path_astar(start_grid, goal_grid)

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
                    if self.occupancy_grid[nr, nc] >= self.forbidden_cost:
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
        
        simplified = world_path[::3]
        if len(world_path) > 0 and world_path[-1] not in simplified:
            simplified.append(world_path[-1])
        
        return simplified

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
