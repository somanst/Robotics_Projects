#!/usr/bin/env python3

import os
import math
import yaml
import heapq
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Range
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped


class AStarDWANode(DTROS):
    """
    Duckiebot assignment node:
    - Creates an 8-connected occupancy grid.
    - Computes a global A* path from A to B.
    - Uses DWA as a local planner to avoid a circular obstacle.
    - Estimates pose using wheel odometry from a chosen reference point (0,0,0).
    - Visualizes the map, global path, obstacle, inflated obstacle, sampled trajectories,
      chosen trajectory, and robot pose using OpenCV.
    """

    def __init__(self, node_name="astar_dwa_node"):
        super(AStarDWANode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        self.vehicle_name = os.environ.get("VEHICLE_NAME", "duckiebot")

        self.load_config()

        # ------------------------------------------------------------------
        # Robot state: reference point is always treated as (0, 0, 0)
        # ------------------------------------------------------------------
        self.x = self.start_x
        self.y = self.start_y
        self.theta = self.start_theta

        self.prev_left_ticks = None
        self.prev_right_ticks = None
        self.left_ticks = None
        self.right_ticks = None

        self.goal_reached = False
        self.path_index = 0

        self.dynamic_obstacles = []

        self.candidate_trajectories = []
        self.best_trajectory = []

        # ------------------------------------------------------------------
        # ROS subscribers and publishers
        # ------------------------------------------------------------------
        left_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        right_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
        wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"

        tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"
        self.sub_tof = rospy.Subscriber(tof_topic, Range, self.tof_callback)

        self.sub_left = rospy.Subscriber(left_topic, WheelEncoderStamped, self.left_encoder_callback)
        self.sub_right = rospy.Subscriber(right_topic, WheelEncoderStamped, self.right_encoder_callback)
        self.cmd_pub = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        # ------------------------------------------------------------------
        # Global path planning
        # ------------------------------------------------------------------
        self.grid_width = int(self.map_width / self.resolution) + 1
        self.grid_height = int(self.map_height / self.resolution) + 1

        self.global_path = self.run_astar(
            (self.start_x, self.start_y),
            (self.goal_x, self.goal_y)
        )

        if not self.global_path:
            rospy.logerr("A* failed to find a path.")
        else:
            rospy.loginfo(f"A* path generated with {len(self.global_path)} waypoints.")
            self.debug_log_global_path()

        # ------------------------------------------------------------------
        # Visualization setup
        # ------------------------------------------------------------------
        self.window_name = "A* + DWA Duckiebot Planner"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        rospy.on_shutdown(self.on_shutdown)

    # ======================================================================
    # Configuration
    # ======================================================================
    def load_config(self):
        project_root = os.getcwd()
        config_path = os.path.join(project_root, "config", "runtime.yaml")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(BASE_DIR, "config", "runtime.yaml")

        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            rospy.loginfo(f"Loaded config from {config_path}")
        else:
            rospy.logwarn(f"No config found at {config_path}. Using defaults.")

        self.tof_threshold = cfg.get("tof_threshold", 0.35)

        # Map
        self.map_width = cfg.get("map_width", 1.5)
        self.map_height = cfg.get("map_height", 1.5)
        self.resolution = cfg.get("resolution", 0.05)

        # Start and goal in world frame
        self.start_x = cfg.get("start_x", 0.0)
        self.start_y = cfg.get("start_y", 0.0)
        self.start_theta = cfg.get("start_theta", 0.0)

        self.goal_x = cfg.get("goal_x", 1.4)
        self.goal_y = cfg.get("goal_y", 1.4)

        # Obstacle
        self.obstacle_x = cfg.get("obstacle_x", 0.75)
        self.obstacle_y = cfg.get("obstacle_y", 0.75)
        self.robot_radius = cfg.get("robot_radius", 0.08)
        self.safety_margin = cfg.get("safety_margin", 0.06)
        self.inflated_radius = self.robot_radius + self.safety_margin

        # Duckiebot physical parameters
        self.wheel_radius = cfg.get("wheel_radius", 0.0318)
        self.wheel_base = cfg.get("wheel_base", 0.10)
        self.encoder_resolution = cfg.get("encoder_resolution", 135)

        # Velocity limits
        self.max_v = cfg.get("max_v", 0.12)
        # Important: if min_v is 0.0, DWA may prefer standing still because
        # the robot is already close to the global path at the beginning.
        self.min_v = cfg.get("min_v", 0.03)
        self.max_w = cfg.get("max_w", 2.5)

        # DWA parameters
        self.dwa_dt = cfg.get("dwa_dt", 0.1)
        self.dwa_horizon = cfg.get("dwa_horizon", 1.2)
        self.v_samples = cfg.get("v_samples", 6)
        self.w_samples = cfg.get("w_samples", 21)

        # DWA cost weights
        self.goal_weight = cfg.get("goal_weight", 1.5)
        self.path_weight = cfg.get("path_weight", 1.0)
        self.heading_weight = cfg.get("heading_weight", 0.5)
        self.obstacle_weight = cfg.get("obstacle_weight", 2.5)
        self.speed_weight = cfg.get("speed_weight", 0.4)
        self.waypoint_weight = cfg.get("waypoint_weight", 2.0)

        # Control thresholds
        self.goal_threshold = cfg.get("goal_threshold", 0.08)
        self.waypoint_threshold = cfg.get("waypoint_threshold", 0.12)

        # Visualization
        self.canvas_size = cfg.get("canvas_size", 900)
        self.margin_px = cfg.get("margin_px", 80)
        self.show_grid = cfg.get("show_grid", True)
        self.grid_color = (55, 55, 55)

        # Debug controls
        self.debug_enabled = cfg.get("debug_enabled", True)
        self.debug_dwa_candidates = cfg.get("debug_dwa_candidates", True)
        self.debug_dwa_rejections = cfg.get("debug_dwa_rejections", True)
        self.debug_trajectory_points = cfg.get("debug_trajectory_points", False)
        self.debug_astar = cfg.get("debug_astar", True)
        self.debug_odometry = cfg.get("debug_odometry", True)
        self.debug_commands = cfg.get("debug_commands", True)
        self.debug_visualization = cfg.get("debug_visualization", False)
        self.debug_throttle_sec = cfg.get("debug_throttle_sec", 1.0)
        self.debug_candidate_limit = cfg.get("debug_candidate_limit", 8)

        self.loop_counter = 0
        self.last_debug_time = 0.0

        rospy.loginfo(
            f"Config: start=({self.start_x:.2f},{self.start_y:.2f}), "
            f"goal=({self.goal_x:.2f},{self.goal_y:.2f}), "
            f"obstacle=({self.obstacle_x:.2f},{self.obstacle_y:.2f}), "
            f"inflated_radius={self.inflated_radius:.2f}"
        )

    # ======================================================================
    # Debug helpers
    # ======================================================================
    def debug_allowed(self):
        if not self.debug_enabled:
            return False
        now = rospy.Time.now().to_sec()
        if now - self.last_debug_time >= self.debug_throttle_sec:
            self.last_debug_time = now
            return True
        return False

    def debug_log_global_path(self):
        if not self.debug_enabled or not self.debug_astar:
            return

        rospy.loginfo("==================== A* GLOBAL PATH DEBUG ====================")
        rospy.loginfo(f"Map size: {self.map_width:.2f}m x {self.map_height:.2f}m")
        rospy.loginfo(f"Grid resolution: {self.resolution:.3f}m")
        rospy.loginfo(f"Grid dimensions: {self.grid_width} x {self.grid_height}")
        rospy.loginfo(f"Start world: ({self.start_x:.3f}, {self.start_y:.3f}, {self.start_theta:.3f})")
        rospy.loginfo(f"Goal world:  ({self.goal_x:.3f}, {self.goal_y:.3f})")
        rospy.loginfo(f"Start grid: {self.world_to_grid(self.start_x, self.start_y)}")
        rospy.loginfo(f"Goal grid:  {self.world_to_grid(self.goal_x, self.goal_y)}")
        rospy.loginfo(f"Number of waypoints: {len(self.global_path)}")

        for i, (x, y) in enumerate(self.global_path):
            rospy.loginfo(f"Waypoint[{i:03d}] = x:{x:.3f}, y:{y:.3f}")

        rospy.loginfo("==============================================================")

    def debug_log_runtime_state(self, v_cmd, w_cmd):
        current_wp = self.get_current_waypoint() if self.global_path else (self.goal_x, self.goal_y)
        goal_dist = math.sqrt((self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2)
        wp_dist = math.sqrt((self.x - current_wp[0]) ** 2 + (self.y - current_wp[1]) ** 2)
        obs_dist = self.distance_to_obstacle(self.x, self.y)

        rospy.loginfo("==================== RUNTIME DEBUG ====================")
        rospy.loginfo(f"Loop counter: {self.loop_counter}")
        rospy.loginfo(f"Robot pose: x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f}")
        rospy.loginfo(f"Current waypoint index: {self.path_index}/{max(len(self.global_path)-1, 0)}")
        rospy.loginfo(f"Current waypoint: x={current_wp[0]:.3f}, y={current_wp[1]:.3f}")
        rospy.loginfo(f"Distance to waypoint: {wp_dist:.3f}")
        rospy.loginfo(f"Distance to goal: {goal_dist:.3f}")
        rospy.loginfo(f"Obstacle center: x={self.obstacle_x:.3f}, y={self.obstacle_y:.3f}")
        rospy.loginfo(f"Distance to obstacle center: {obs_dist:.3f}")
        rospy.loginfo(f"Inflated obstacle radius: {self.inflated_radius:.3f}")
        rospy.loginfo(f"Command selected: v={v_cmd:.3f}, w={w_cmd:.3f}")
        rospy.loginfo(f"Goal reached: {self.goal_reached}")
        rospy.loginfo("=======================================================")

    def debug_log_dwa_summary(self, best_v, best_w, best_cost, total_samples, rejected_samples):
        rospy.loginfo("-------------------- DWA DEBUG --------------------")
        rospy.loginfo(f"DWA dt={self.dwa_dt:.3f}, horizon={self.dwa_horizon:.3f}")
        rospy.loginfo(f"Samples: v={self.v_samples}, w={self.w_samples}, total={total_samples}")
        rospy.loginfo(f"Rejected samples: {rejected_samples}")
        rospy.loginfo(f"Accepted samples: {total_samples - rejected_samples}")
        rospy.loginfo(f"Best cost: {best_cost:.4f}")
        rospy.loginfo(f"Best command: v={best_v:.3f}, w={best_w:.3f}")

        if self.best_trajectory:
            start = self.best_trajectory[0]
            end = self.best_trajectory[-1]
            rospy.loginfo(f"Best trajectory start: x={start[0]:.3f}, y={start[1]:.3f}, theta={start[2]:.3f}")
            rospy.loginfo(f"Best trajectory end:   x={end[0]:.3f}, y={end[1]:.3f}, theta={end[2]:.3f}")

            if self.debug_trajectory_points:
                for i, point in enumerate(self.best_trajectory):
                    rospy.loginfo(f"BestTraj[{i:02d}] x={point[0]:.3f}, y={point[1]:.3f}, theta={point[2]:.3f}")
        else:
            rospy.logwarn("No valid best trajectory found. Robot will stop.")

        rospy.loginfo("---------------------------------------------------")

    def debug_log_candidate(self, idx, v, w, cost, trajectory, rejected, reject_reason):
        if not self.debug_dwa_candidates:
            return
        if idx >= self.debug_candidate_limit:
            return

        status = "REJECTED" if rejected else "ACCEPTED"
        end = trajectory[-1] if trajectory else (self.x, self.y, self.theta)
        rospy.loginfo(
            f"DWA Candidate[{idx:03d}] {status}: v={v:.3f}, w={w:.3f}, "
            f"cost={cost:.4f}, end=({end[0]:.3f},{end[1]:.3f},{end[2]:.3f}), reason={reject_reason}"
        )

        if self.debug_trajectory_points:
            for j, point in enumerate(trajectory):
                rospy.loginfo(f"  Cand[{idx:03d}].Traj[{j:02d}] x={point[0]:.3f}, y={point[1]:.3f}, theta={point[2]:.3f}")

    # ======================================================================
    # Encoder callbacks and odometry
    # ======================================================================
    def left_encoder_callback(self, msg):
        self.left_ticks = msg.data

    def right_encoder_callback(self, msg):
        self.right_ticks = msg.data

    def tof_callback(self, msg):
        """
        Callback for the Time-of-Flight (ToF) sensor.
        Detects dynamic obstacles and registers their global positions.
        """
        d = msg.range
        
        # Check if the reading is valid and within the threshold
        if 0.0 < d < self.tof_threshold:
            # Add an offset since the sensor is mounted on the front bumper (e.g., 5 cm from center)
            d += 0.05 
            
            # Calculate the global (x, y) coordinates of the detected obstacle
            obs_x = self.x + d * math.cos(self.theta)
            obs_y = self.y + d * math.sin(self.theta)

            is_new = True

            # --- YENI EKLENEN KISIM: Statik Engel Kontrolü ---
            # Eger ToF'un gordugu nokta, zaten bildigimiz statik engele 20 cm'den yakinsa, bunu yeni sayma!
            dist_to_static = math.sqrt((obs_x - self.obstacle_x)**2 + (obs_y - self.obstacle_y)**2)
            if dist_to_static < 0.20:
                is_new = False
            # --------------------------------------------------

            # Clustering: Prevent adding the same dynamic obstacle multiple times
            if is_new:
                for (ex, ey) in self.dynamic_obstacles:
                    # If an obstacle is already recorded within a 15 cm radius, ignore this reading
                    if math.sqrt((obs_x - ex)**2 + (obs_y - ey)**2) < 0.15:
                        is_new = False
                        break

            # If it is a newly detected obstacle, add it to the map
            if is_new:
                self.dynamic_obstacles.append((obs_x, obs_y))
                rospy.logwarn(f"NEW OBSTACLE DETECTED: x={obs_x:.2f}, y={obs_y:.2f} (Distance: {d:.2f}m)")

                
    def update_odometry(self):
        if self.left_ticks is None or self.right_ticks is None:
            if self.debug_enabled and self.debug_odometry and self.debug_allowed():
                rospy.logwarn("Odometry waiting for encoder ticks...")
            return

        if self.prev_left_ticks is None or self.prev_right_ticks is None:
            self.prev_left_ticks = self.left_ticks
            self.prev_right_ticks = self.right_ticks
            if self.debug_enabled and self.debug_odometry:
                rospy.loginfo(f"Odometry initialized: left_ticks={self.left_ticks}, right_ticks={self.right_ticks}")
            return

        delta_left = self.left_ticks - self.prev_left_ticks
        delta_right = self.right_ticks - self.prev_right_ticks

        self.prev_left_ticks = self.left_ticks
        self.prev_right_ticks = self.right_ticks

        dL = 2.0 * math.pi * self.wheel_radius * (delta_left / self.encoder_resolution)
        dR = 2.0 * math.pi * self.wheel_radius * (delta_right / self.encoder_resolution)

        old_x, old_y, old_theta = self.x, self.y, self.theta

        d_center = 0.5 * (dL + dR)
        d_theta = (dR - dL) / self.wheel_base

        if abs(d_theta) < 1e-6:
            self.x += d_center * math.cos(self.theta)
            self.y += d_center * math.sin(self.theta)
        else:
            radius = d_center / d_theta
            self.x += radius * (math.sin(self.theta + d_theta) - math.sin(self.theta))
            self.y -= radius * (math.cos(self.theta + d_theta) - math.cos(self.theta))
            self.theta = self.normalize_angle(self.theta + d_theta)

        if self.debug_enabled and self.debug_odometry and self.debug_allowed():
            rospy.loginfo("-------------------- ODOMETRY DEBUG --------------------")
            rospy.loginfo(f"Ticks: left={self.left_ticks}, right={self.right_ticks}")
            rospy.loginfo(f"Delta ticks: left={delta_left}, right={delta_right}")
            rospy.loginfo(f"Wheel distances: dL={dL:.5f}, dR={dR:.5f}")
            rospy.loginfo(f"d_center={d_center:.5f}, d_theta={d_theta:.5f}")
            rospy.loginfo(f"Old pose: x={old_x:.3f}, y={old_y:.3f}, theta={old_theta:.3f}")
            rospy.loginfo(f"New pose: x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f}")
            rospy.loginfo("--------------------------------------------------------")

    # ======================================================================
    # A* global planner
    # ======================================================================
    def world_to_grid(self, x, y):
        gx = int(round(x / self.resolution))
        gy = int(round(y / self.resolution))
        gx = max(0, min(self.grid_width - 1, gx))
        gy = max(0, min(self.grid_height - 1, gy))
        return gx, gy

    def grid_to_world(self, gx, gy):
        return gx * self.resolution, gy * self.resolution

    def heuristic(self, a, b):
        ax, ay = a
        bx, by = b
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def get_neighbors(self, cell):
        x, y = cell
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        neighbors = []
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                # Convert grid cell to world coordinates to check obstacle clearance
                wx, wy = self.grid_to_world(nx, ny)
                if self.distance_to_obstacle(wx, wy) <= self.inflated_radius:
                    continue # Skip this neighbor, it's inside the inflated obstacle!

                cost = math.sqrt(dx * dx + dy * dy)
                neighbors.append(((nx, ny), cost))
        return neighbors

    def run_astar(self, start_world, goal_world):
        start = self.world_to_grid(start_world[0], start_world[1])
        goal = self.world_to_grid(goal_world[0], goal_world[1])

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {start: None}
        g_cost = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_cost[current] + move_cost

                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    f_cost = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_cost, neighbor))
                    came_from[neighbor] = current

        return []

    def reconstruct_path(self, came_from, current):
        path_grid = []
        while current is not None:
            path_grid.append(current)
            current = came_from[current]
        path_grid.reverse()

        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
        return path_world

    # ======================================================================
    # DWA local planner
    # ======================================================================
    def rollout(self, v, w):
        x = self.x
        y = self.y
        theta = self.theta
        trajectory = []

        steps = int(self.dwa_horizon / self.dwa_dt)

        for _ in range(steps):
            x += v * math.cos(theta) * self.dwa_dt
            y += v * math.sin(theta) * self.dwa_dt
            theta = self.normalize_angle(theta + w * self.dwa_dt)
            trajectory.append((x, y, theta))

        return trajectory

    def distance_to_obstacle(self, x, y):
            """
            Calculates the minimum distance from the given (x,y) point 
            to all known obstacles (both static and dynamic).
            """
            # Distance to the known static obstacle
            min_dist = math.sqrt((x - self.obstacle_x) ** 2 + (y - self.obstacle_y) ** 2)
            
            # Check distances to all dynamically detected obstacles
            for (ox, oy) in self.dynamic_obstacles:
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    
            return min_dist

    def trajectory_collision_reason(self, trajectory):
        for i, (x, y, _) in enumerate(trajectory):
            obs_dist = self.distance_to_obstacle(x, y)
            if obs_dist <= self.inflated_radius:
                return True, f"obstacle_collision at step={i}, obs_dist={obs_dist:.3f}"
            if x < 0.0 or x > self.map_width or y < 0.0 or y > self.map_height:
                return True, f"outside_map at step={i}, x={x:.3f}, y={y:.3f}"
        return False, "safe"

    def trajectory_collision(self, trajectory):
        collision, _ = self.trajectory_collision_reason(trajectory)
        return collision

    def distance_to_global_path(self, x, y):
        if not self.global_path:
            return 0.0
        return min(math.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in self.global_path)

    def get_current_waypoint(self):
        if not self.global_path:
            if self.debug_enabled:
                rospy.logwarn("No global path available. Using final goal as waypoint.")
            return self.goal_x, self.goal_y

        while self.path_index < len(self.global_path) - 1:
            wx, wy = self.global_path[self.path_index]
            dist = math.sqrt((self.x - wx) ** 2 + (self.y - wy) ** 2)
            if dist > self.waypoint_threshold:
                break

            old_index = self.path_index
            self.path_index += 1
            if self.debug_enabled:
                rospy.loginfo(
                    f"Waypoint advanced: {old_index} -> {self.path_index}, "
                    f"previous waypoint distance={dist:.3f}, threshold={self.waypoint_threshold:.3f}"
                )

        return self.global_path[self.path_index]

    def score_trajectory(self, trajectory, v):
        if not trajectory:
            return float("inf")

        end_x, end_y, end_theta = trajectory[-1]

        # Main final goal cost
        goal_dist = math.sqrt((end_x - self.goal_x) ** 2 + (end_y - self.goal_y) ** 2)

        # Stay close to the A* path
        path_dist = self.distance_to_global_path(end_x, end_y)

        # Prefer moving toward the next waypoint, not just staying on the path
        wx, wy = self.get_current_waypoint()
        waypoint_dist = math.sqrt((end_x - wx) ** 2 + (end_y - wy) ** 2)

        desired_heading = math.atan2(wy - end_y, wx - end_x)
        heading_error = abs(self.normalize_angle(desired_heading - end_theta))

        # Penalize getting close to the inflated obstacle
        min_obstacle_dist = min(self.distance_to_obstacle(x, y) for x, y, _ in trajectory)
        clearance = max(min_obstacle_dist - self.inflated_radius, 0.001)
        obstacle_cost = 1.0 / clearance

        # Stronger preference for actual forward motion
        speed_cost = self.max_v - v

        total = (
            self.goal_weight * goal_dist +
            self.waypoint_weight * waypoint_dist +
            self.path_weight * path_dist +
            self.heading_weight * heading_error +
            self.obstacle_weight * obstacle_cost +
            self.speed_weight * speed_cost
        )

        return total

    def dwa_control(self):
        self.candidate_trajectories = []
        self.best_trajectory = []

        best_cost = float("inf")
        best_v = 0.0
        best_w = 0.0

        v_values = np.linspace(self.min_v, self.max_v, self.v_samples)
        w_values = np.linspace(-self.max_w, self.max_w, self.w_samples)

        total_samples = 0
        rejected_samples = 0
        candidate_index = 0
        should_print_dwa = self.debug_enabled and self.debug_allowed()

        for v in v_values:
            for w in w_values:
                total_samples += 1
                trajectory = self.rollout(v, w)
                self.candidate_trajectories.append(trajectory)

                rejected, reject_reason = self.trajectory_collision_reason(trajectory)

                if rejected:
                    rejected_samples += 1
                    if should_print_dwa and self.debug_dwa_rejections:
                        self.debug_log_candidate(candidate_index, v, w, float("inf"), trajectory, True, reject_reason)
                    candidate_index += 1
                    continue

                cost = self.score_trajectory(trajectory, v)

                if should_print_dwa:
                    self.debug_log_candidate(candidate_index, v, w, cost, trajectory, False, reject_reason)

                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w
                    self.best_trajectory = trajectory

                candidate_index += 1

        if should_print_dwa:
            self.debug_log_dwa_summary(best_v, best_w, best_cost, total_samples, rejected_samples)

        # --------------------------------------------------
        # Recovery behavior:
        # If no valid trajectory found, back up and turn
        # --------------------------------------------------
        if best_cost == float("inf"):
            rospy.logwarn("No valid forward DWA trajectory. Executing recovery backup.")

            reverse_v = -0.05

            # Turn away from obstacle
            dx = self.obstacle_x - self.x
            dy = self.obstacle_y - self.y
            obs_angle = math.atan2(dy, dx)

            angle_diff = self.normalize_angle(obs_angle - self.theta)

            if angle_diff > 0:
                reverse_w = -1.0
            else:
                reverse_w = 1.0

            recovery_traj = self.rollout(reverse_v, reverse_w)

            self.best_trajectory = recovery_traj

            return reverse_v, reverse_w


        return best_v, best_w

    # ======================================================================
    # Robot control
    # ======================================================================
    def publish_cmd(self, v, w):
        vel_left_raw = v - (self.wheel_base / 2.0) * w
        vel_right_raw = v + (self.wheel_base / 2.0) * w

        vel_left = max(-1.0, min(1.0, vel_left_raw))
        vel_right = max(-1.0, min(1.0, vel_right_raw))

        if self.debug_enabled and self.debug_commands and self.debug_allowed():
            rospy.loginfo("-------------------- COMMAND DEBUG --------------------")
            rospy.loginfo(f"Input command: v={v:.3f}, w={w:.3f}")
            rospy.loginfo(f"Raw wheel cmd: left={vel_left_raw:.3f}, right={vel_right_raw:.3f}")
            rospy.loginfo(f"Clamped wheel cmd: left={vel_left:.3f}, right={vel_right:.3f}")
            rospy.loginfo("-------------------------------------------------------")

        msg = WheelsCmdStamped()
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def check_goal_reached(self):
        dist = math.sqrt((self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2)
        if self.debug_enabled and self.debug_allowed():
            rospy.loginfo(f"Goal check: distance={dist:.3f}, threshold={self.goal_threshold:.3f}")

        if dist <= self.goal_threshold:
            self.goal_reached = True
            self.stop_robot()
            rospy.loginfo("Goal reached.")

    # ======================================================================
    # Visualization
    # ======================================================================
    def world_to_pixel(self, x, y):
        usable = self.canvas_size - 2 * self.margin_px
        scale_x = usable / self.map_width
        scale_y = usable / self.map_height
        scale = min(scale_x, scale_y)

        px = int(self.margin_px + x * scale)
        py = int(self.canvas_size - self.margin_px - y * scale)
        return px, py

    def draw_trajectory(self, canvas, trajectory, color, thickness):
        if len(trajectory) < 2:
            return

        for i in range(len(trajectory) - 1):
            x1, y1, _ = trajectory[i]
            x2, y2, _ = trajectory[i + 1]
            p1 = self.world_to_pixel(x1, y1)
            p2 = self.world_to_pixel(x2, y2)
            cv2.line(canvas, p1, p2, color, thickness)

    def draw_grid(self, canvas):
        if not self.show_grid:
            return

        # Draw occupancy-grid style cells using the same resolution as A*
        for gx in range(self.grid_width):
            x = gx * self.resolution
            p1 = self.world_to_pixel(x, 0.0)
            p2 = self.world_to_pixel(x, self.map_height)
            cv2.line(canvas, p1, p2, self.grid_color, 1)

        for gy in range(self.grid_height):
            y = gy * self.resolution
            p1 = self.world_to_pixel(0.0, y)
            p2 = self.world_to_pixel(self.map_width, y)
            cv2.line(canvas, p1, p2, self.grid_color, 1)

    def draw_visualization(self):
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)

        self.draw_grid(canvas)

        # Border/map frame
        p00 = self.world_to_pixel(0.0, 0.0)
        p11 = self.world_to_pixel(self.map_width, self.map_height)
        cv2.rectangle(canvas, (p00[0], p11[1]), (p11[0], p00[1]), (180, 180, 180), 2)

        # Global A* path
        for i in range(len(self.global_path) - 1):
            p1 = self.world_to_pixel(*self.global_path[i])
            p2 = self.world_to_pixel(*self.global_path[i + 1])
            cv2.line(canvas, p1, p2, (0, 0, 255), 2)

        # Start and goal
        sx, sy = self.world_to_pixel(self.start_x, self.start_y)
        gx, gy = self.world_to_pixel(self.goal_x, self.goal_y)
        cv2.circle(canvas, (sx, sy), 8, (255, 0, 0), -1)
        cv2.putText(canvas, "A", (sx + 8, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(canvas, (gx, gy), 8, (0, 255, 0), -1)
        cv2.putText(canvas, "B", (gx + 8, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Obstacle and inflated boundary
        ox, oy = self.world_to_pixel(self.obstacle_x, self.obstacle_y)
        inflated_px = int(self.inflated_radius * ((self.canvas_size - 2 * self.margin_px) / self.map_width))
        cv2.circle(canvas, (ox, oy), inflated_px, (0, 0, 255), 2)
        cv2.circle(canvas, (ox, oy), 6, (0, 0, 255), -1)

        # Draw dynamic (ToF detected) obstacles in orange
        for (ox, oy) in self.dynamic_obstacles:
            px, py = self.world_to_pixel(ox, oy)
            
            # Inflated safety boundary
            cv2.circle(canvas, (px, py), inflated_px, (0, 165, 255), 2)
            # Center of the obstacle
            cv2.circle(canvas, (px, py), 6, (0, 165, 255), -1)

        # DWA candidate trajectories
        for traj in self.candidate_trajectories:
            self.draw_trajectory(canvas, traj, (70, 70, 70), 1)

        # Best DWA trajectory
        self.draw_trajectory(canvas, self.best_trajectory, (0, 255, 255), 2)

        # Robot and sensing area
        rx, ry = self.world_to_pixel(self.x, self.y)
        robot_radius_px = max(4, int(self.robot_radius * ((self.canvas_size - 2 * self.margin_px) / self.map_width)))
        cv2.circle(canvas, (rx, ry), robot_radius_px, (255, 255, 0), 2)

        sensing_radius = 0.08
        sensing_px = int(sensing_radius * ((self.canvas_size - 2 * self.margin_px) / self.map_width))
        cv2.circle(canvas, (rx, ry), sensing_px, (80, 80, 180), 1)

        arrow_len = 0.12
        hx = self.x + arrow_len * math.cos(self.theta)
        hy = self.y + arrow_len * math.sin(self.theta)
        hpx, hpy = self.world_to_pixel(hx, hy)
        cv2.arrowedLine(canvas, (rx, ry), (hpx, hpy), (255, 255, 0), 2, tipLength=0.3)

        # Text status
        cv2.putText(canvas, f"x={self.x:.2f} y={self.y:.2f} theta={self.theta:.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, f"waypoint {self.path_index + 1}/{len(self.global_path)}",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    # ======================================================================
    # Utilities
    # ======================================================================
    def normalize_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # ======================================================================
    # Main loop
    # ======================================================================
    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            self.update_odometry()
            self.check_goal_reached()

            v, w = 0.0, 0.0

            if not self.goal_reached:
                v, w = self.dwa_control()
                self.publish_cmd(v, w)

            if self.debug_enabled and self.debug_allowed():
                self.debug_log_runtime_state(v, w)

            self.draw_visualization()
            self.loop_counter += 1
            rate.sleep()

    def on_shutdown(self):
        rospy.loginfo("Stopping robot and closing windows.")
        self.stop_robot()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = AStarDWANode()
    node.run()


"""
Example runtime.yaml:

map_width: 1.5
map_height: 1.5
resolution: 0.05

start_x: 0.0
start_y: 0.0
start_theta: 0.0

goal_x: 1.4
goal_y: 1.4

obstacle_x: 0.75
obstacle_y: 0.75

robot_radius: 0.08
safety_margin: 0.06

wheel_radius: 0.0318
wheel_base: 0.10
encoder_resolution: 135

max_v: 0.12
min_v: 0.03
max_w: 2.5

dwa_dt: 0.1
dwa_horizon: 1.2
v_samples: 6
w_samples: 21

goal_weight: 1.5
path_weight: 1.0
heading_weight: 0.5
obstacle_weight: 2.5
speed_weight: 0.4
waypoint_weight: 2.0

goal_threshold: 0.08
waypoint_threshold: 0.12

canvas_size: 900
margin_px: 80
show_grid: true

# Debug options
debug_enabled: true
debug_dwa_candidates: true
debug_dwa_rejections: true
debug_trajectory_points: false
debug_astar: true
debug_odometry: true
debug_commands: true
debug_visualization: false
debug_throttle_sec: 1.0
debug_candidate_limit: 8
"""
