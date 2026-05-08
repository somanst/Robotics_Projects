#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge

from duckietown_msgs.msg import WheelEncoderStamped
from sensor_msgs.msg import CameraInfo

import numpy as np

from duckietown_msgs.msg import WheelsCmdStamped
import math

import yaml

NODES = {
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1), 7: (3, 1),
    8: (0, 2), 9: (1, 2), 10: (2, 2), 11: (3, 2),
    12: (0, 3), 13: (1, 3), 14: (2, 3), 15: (3, 3)
}

# Valid Edges and Path Costs
EDGES = {
    0: {1: 1.5, 4: 2.0},
    1: {0: 1.5, 2: 1.0, 5: 2.0},
    2: {1: 1.0, 3: 1.0, 6: 1.5},
    3: {2: 1.0},
    4: {0: 2.0, 8: 1.5},
    5: {1: 2.0, 6: 1.0, 9: 2.0},
    6: {2: 1.5, 5: 1.0, 7: 0.5, 10: 4.0},
    7: {6: 0.5, 11: 1.5},
    8: {4: 1.5, 9: 1.5, 12: 2.0},
    9: {5: 2.0, 8: 1.5, 10: 2.0},
    10: {6: 4.0, 9: 2.0, 11: 1.0, 14: 1.5},
    11: {7: 1.5, 10: 1.0},
    12: {8: 2.0, 13: 1.5},
    13: {12: 1.5, 14: 2.0},
    14: {10: 1.5, 13: 2.0, 15: 1.0},
    15: {14: 1.0}
}

class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        

        super(CameraReaderNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.VISUALIZATION
        )

        self.load_runtime_config()

        self.latest_image = None
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        
        self._window = "Localizer"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        self.map_w = 1000
        self.map_h = 1000

        # pixels per meter
        self.scale = 400

        # screen margin
        self.margin = 150

        # world origin on screen
        self.origin_px = (self.margin, self.map_h - self.margin)
        
        self.base_map = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.map_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8) 
        cv2.namedWindow("Top-Down Map", cv2.WINDOW_AUTOSIZE)

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.camera_info_callback
        )

        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._ticks_left = None
        self._ticks_right = None
        self.sub_left = rospy.Subscriber(
            self._left_encoder_topic,
            WheelEncoderStamped,
            self.callback_left
        )
        self.sub_right = rospy.Subscriber(
            self._right_encoder_topic,
            WheelEncoderStamped,
            self.callback_right
        )

        self._prev_ticks_left = None
        self._prev_ticks_right = None

        grid_step = 0.5

        self.tagMap = {
            0:  np.array([0 * grid_step, 0 * grid_step, 0.0]),
            1:  np.array([1 * grid_step, 0 * grid_step, 0.0]),
            2:  np.array([2 * grid_step, 0 * grid_step, 0.0]),
            3:  np.array([3 * grid_step, 0 * grid_step, 0.0]),

            4:  np.array([0 * grid_step, 1 * grid_step, 0.0]),
            5:  np.array([1 * grid_step, 1 * grid_step, 0.0]),
            6:  np.array([2 * grid_step, 1 * grid_step, 0.0]),
            7:  np.array([3 * grid_step, 1 * grid_step, 0.0]),

            8:  np.array([0 * grid_step, 2 * grid_step, 0.0]),
            9:  np.array([1 * grid_step, 2 * grid_step, 0.0]),
            10: np.array([2 * grid_step, 2 * grid_step, 0.0]),
            11: np.array([3 * grid_step, 2 * grid_step, 0.0]),

            12: np.array([0 * grid_step, 3 * grid_step, 0.0]),
            13: np.array([1 * grid_step, 3 * grid_step, 0.0]),
            14: np.array([2 * grid_step, 3 * grid_step, 0.0]),
            15: np.array([3 * grid_step, 3 * grid_step, 0.0]),
        }
        self.edges = [
            (0,1), (1,2), (2,3),
            (0,4), (1,5), (2,6), (6,7),
            (4,8), (8,9), (9,10), (10,11),
            (8,12), (12,13), (13,14), (14,15),
            (5,6), (6,10), (10,14)
        ]
        self.localizationState = False
        self.draw_fixed_tags()

        self.K = None
        self.D = None

        rospy.on_shutdown(self.on_shutdown)

        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        self.cmd_pub = rospy.Publisher(
            wheels_topic,
            WheelsCmdStamped,
            queue_size=1
        )

        # self.start_node = rospy.get_param("~start_node", 0)
        # self.goal_node = rospy.get_param("~goal_node", 15)

        self.path = []
        self.path_index = 0
        self.goal_reached = False

        self.path, total_cost = self.calculate_a_star(self.start_node, self.goal_node)

        self.draw_planned_path()
        
        rospy.loginfo(f"Start node: N{self.start_node}")
        rospy.loginfo(f"Goal node: N{self.goal_node}")
        print("""
              ########################################################
              A* CALCULATION
              """)
        path_str = " -> ".join([f"N{node}" for node in self.path])
        rospy.loginfo(f"Calculated Path: {path_str}")
        rospy.loginfo(f"""
                      Total Path Cost: {total_cost:.2f}
              ######################################################### 
               """)
        print("""
            NAVIGATION:
              
                """)

        self.x = self.tagMap[self.start_node][0]
        self.y = self.tagMap[self.start_node][1]
        self.theta = self.tagMap[self.start_node][2]

        # self.node_reach_threshold = rospy.get_param("~node_reach_threshold", 0.25)   # 10 cm as you requested
        # self.angle_tolerance = rospy.get_param("~angle_tolerance", 0.3)       # about 7 deg
        # self.max_v = rospy.get_param("~max_v", 0.11)
        # self.max_w = rospy.get_param("~max_w", 2.0)
        # self.search_w = rospy.get_param("~search_w", 1.5)

        self.last_tag_time = rospy.Time.now()
        # self.tag_timeout = rospy.get_param("~tag_timeout", 2.0)   # if no tag seen for 0.5 sec => lost

    def load_runtime_config(self):
        project_root = os.getcwd()
        config_path = os.path.join(project_root, "config", "runtime.yaml")

        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            rospy.loginfo(f"Loaded runtime config from: {config_path}")
        else:
            rospy.logwarn(f"Config file not found: {config_path}. Using defaults.")

        self.start_node = cfg.get("start_node", 0)
        self.goal_node = cfg.get("goal_node", 15)

        self.max_v = cfg.get("max_v", 0.11)
        self.max_w = cfg.get("max_w", 2.0)
        self.angle_tolerance = cfg.get("angle_tolerance", 0.3)
        self.node_reach_threshold = cfg.get("node_reach_threshold", 0.1)
        self.tag_timeout = cfg.get("tag_timeout", 2.0)
        self.camera_offset = cfg.get("camera_offset", 0.15)

        self.search_rotate_seconds = cfg.get("search_rotate_seconds", 1.0)
        self.search_stop_seconds = cfg.get("search_stop_seconds", 0.5)
        self.search_w = cfg.get("search_w", 1.5)

        rospy.loginfo(
            f"Config: start=N{self.start_node}, goal=N{self.goal_node}, "
            f"max_v={self.max_v}, max_w={self.max_w}, "
            f"angle_tol={self.angle_tolerance}, reach_th={self.node_reach_threshold}, "
            f"tag_timeout={self.tag_timeout}"
        )

    def calculate_a_star(self, start, goal):
        def heuristic(n1, n2):
            # Manhattan distance heuristic
            return abs(NODES[n1][0] - NODES[n2][0]) + abs(NODES[n1][1] - NODES[n2][1])
            
        open_list = {start}
        closed_list = set()
        g_costs = {start: 0.0}
        f_costs = {start: heuristic(start, goal)}
        parents = {start: None}

        while open_list:
            current = None
            for node in open_list:
                # Select the node with the lowest f_cost. Tie-breaker applies to h_cost.
                if current is None or f_costs[node] < f_costs[current] or \
                   (f_costs[node] == f_costs[current] and heuristic(node, goal) < heuristic(current, goal)):
                    current = node

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parents[current]
                path.reverse()
                return path, g_costs[goal]

            open_list.remove(current)
            closed_list.add(current)

            for neighbor, cost in EDGES.get(current, {}).items():
                if neighbor in closed_list: continue
                tentative_g = g_costs[current] + cost
                
                if neighbor not in open_list:
                    open_list.add(neighbor)
                elif tentative_g >= g_costs.get(neighbor, float('inf')):
                    continue # Not a better path
                    
                parents[neighbor] = current
                g_costs[neighbor] = tentative_g
                f_costs[neighbor] = tentative_g + heuristic(neighbor, goal)
        return [], 0.0

    def stop_robot(self):
        msg = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self.cmd_pub.publish(msg)

    def publish_cmd(self, v, w):
        L = 0.1  # wheelbase (same as your odometry)

        # convert (v, w) → wheel speeds
        v_left = v - (L / 2.0) * w
        v_right = v + (L / 2.0) * w

        # clamp (VERY IMPORTANT for Duckietown)
        v_left = max(-1.0, min(1.0, v_left))
        v_right = max(-1.0, min(1.0, v_right))

        msg = WheelsCmdStamped(
            vel_left=v_left,
            vel_right=v_right
        )

        self.cmd_pub.publish(msg)

    def get_current_target_node(self):
        if self.path_index >= len(self.path) - 1:
            return None
        return self.path[self.path_index + 1]

    def distance_to_target(self, target_id):
        tx, ty, _ = self.tagMap[target_id]
        dx = tx - self.x
        dy = ty - self.y
        return math.sqrt(dx*dx + dy*dy)

    def heading_to_target(self, target_id):
        tx, ty, _ = self.tagMap[target_id]
        dx = tx - self.x
        dy = ty - self.y
        return math.atan2(dy, dx)

    def advance_target_if_reached(self):
        target_id = self.get_current_target_node()
        if target_id is None:
            return

        dist = self.distance_to_target(target_id)

        if dist < self.node_reach_threshold:
            rospy.loginfo(f"Reached node N{target_id}")
            self.path_index += 1

            if self.path_index >= len(self.path) - 1:
                self.goal_reached = True
                self.stop_robot()
                rospy.loginfo("Goal Reached")

    def navigate_to_current_target(self):
        if self.goal_reached:
            self.stop_robot()
            return

        target_id = self.get_current_target_node()
        if target_id is None:
            self.goal_reached = True
            self.stop_robot()
            rospy.loginfo("Goal Reached")
            return

        # Check if target already reached
        self.advance_target_if_reached()
        if self.goal_reached:
            return

        target_id = self.get_current_target_node()
        tx, ty, _ = self.tagMap[target_id]

        dx = tx - self.x
        dy = ty - self.y
        distance = math.sqrt(dx*dx + dy*dy)

        desired_theta = math.atan2(dy, dx)
        heading_error = self.normalize(desired_theta - self.theta)

        # 1) Turn first if heading error is large
        if abs(heading_error) > self.angle_tolerance:
            w = 5.0 * heading_error
            w = max(-self.max_w, min(self.max_w, w))
            self.publish_cmd(0.0, w)
            return

        # 2) Move forward if nearly aligned
        v = 0.8 * distance
        v = max(0.0, min(self.max_v, v))

        # small heading correction while moving
        w = 1.5 * heading_error
        w = max(-self.max_w, min(self.max_w, w))

        self.publish_cmd(v, w)

    def world_to_pixel(self, x, y):
        px = int(self.origin_px[0] + x * self.scale)
        py = int(self.origin_px[1] - y * self.scale)
        return px, py   

    def draw_fixed_tags(self):
        self.base_map[:] = 0

        for tag_id, pose in self.tagMap.items():
            tx, ty, ttheta = pose
            px, py = self.world_to_pixel(tx, ty)

            cv2.circle(self.base_map, (px, py), 12, (255, 0, 0), -1)
            cv2.putText(self.base_map, f"N{tag_id}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            arrow_len_m = 0.08
            ex = tx + arrow_len_m * np.cos(ttheta)
            ey = ty + arrow_len_m * np.sin(ttheta)
            epx, epy = self.world_to_pixel(ex, ey)

            cv2.arrowedLine(self.base_map, (px, py), (epx, epy),
                            (255, 255, 0), 2, tipLength=0.3)
        
            for a, b in self.edges:
                ax, ay, _ = self.tagMap[a]
                bx, by, _ = self.tagMap[b]
                apx, apy = self.world_to_pixel(ax, ay)
                bpx, bpy = self.world_to_pixel(bx, by)
                cv2.line(self.base_map, (apx, apy), (bpx, bpy), (120, 120, 120), 2)

    def draw_planned_path(self):
        if len(self.path) < 2:
            return

        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]

            ax, ay, _ = self.tagMap[a]
            bx, by, _ = self.tagMap[b]

            apx, apy = self.world_to_pixel(ax, ay)
            bpx, bpy = self.world_to_pixel(bx, by)

            # draw chosen path in red
            cv2.line(self.base_map, (apx, apy), (bpx, bpy), (0, 0, 255), 4)

        # optional: highlight nodes on the chosen path
        for node_id in self.path:
            x, y, _ = self.tagMap[node_id]
            px, py = self.world_to_pixel(x, y)
            cv2.circle(self.base_map, (px, py), 14, (0, 0, 255), 2)

    def draw_on_map(self):
        self.map_canvas = self.base_map.copy()

        px, py = self.world_to_pixel(self.x, self.y)

        if 0 <= px < self.map_w and 0 <= py < self.map_h:
            color = (0, 255, 0) if self.localizationState else (0, 0, 255)

            cv2.circle(self.map_canvas, (px, py), 6, color, -1)

            arrow_len_m = 0.10  # 10 cm arrow in world space
            end_x = self.x + arrow_len_m * np.cos(self.theta)
            end_y = self.y + arrow_len_m * np.sin(self.theta)
            ex, ey = self.world_to_pixel(end_x, end_y)

            cv2.arrowedLine(self.map_canvas, (px, py), (ex, ey), color, 2, tipLength=0.3)

        cv2.imshow("Top-Down Map", self.map_canvas)

    def callback(self, msg):
        self.latest_image = msg 

    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data

    def camera_info_callback(self, msg):
        self.K = np.array(msg.K).reshape((3, 3))
        self.D = np.array(msg.D)

    def normalize(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def detectMarkers(self, image):
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        arucoParams = cv2.aruco.DetectorParameters_create()

        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeMax = 35
        arucoParams.minMarkerPerimeterRate = 0.03

        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        return corners, ids, rejected

    def getPose(self, image, corners, ids):
        cameraMatrix, distCoeffs = self.K, self.D
        markerLength = 0.065

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, markerLength, cameraMatrix, distCoeffs
        )

        #xSum, ySum, angles = 0, 0, []

        lowestDist, angles = 99999999, []
        selectedX, selectedY = 0,0
        for i in range(len(ids)):
            rvec, tvec = rvecs[i], tvecs[i]
            tag_id = ids[i][0]

            cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            R, _ = cv2.Rodrigues(rvec)
            yaw = np.arctan2(R[-1,0], R[0,0])


            local_x = tvec[0][2]
            local_y = -tvec[0][0]


            world_dx = local_x * np.cos(self.theta) - local_y * np.sin(self.theta)
            world_dy = local_x * np.sin(self.theta) + local_y * np.cos(self.theta)

            if tag_id not in self.tagMap:
                tag_theta = self.theta + yaw
                self.tagMap[tag_id] = np.array([self.x + world_dx, self.y + world_dy, tag_theta])
                
                mx, my, t = self.tagMap[tag_id]
                px = int(500 - my * 100)
                py = int(500 - mx * 100)

                cv2.circle(self.base_map, (px, py), 15, (255, 0, 0), -1)


            # xSum += self.tagMap[tag_id][0] - world_dx
            # ySum += self.tagMap[tag_id][1] - world_dy

            if (local_x + local_y) < lowestDist:
                lowestDist = local_x + local_y
                selectedX = self.tagMap[tag_id][0] - world_dx
                selectedY = self.tagMap[tag_id][1] - world_dy

                estimatedTheta = self.normalize(self.tagMap[tag_id][2] - yaw)
                

            
            # estimatedTheta = self.normalize(self.tagMap[tag_id][2] - yaw)
            # angles.append(estimatedTheta)
        # self.x = xSum/len(ids)
        # self.y = ySum/len(ids)

        angles.append(estimatedTheta)

        xTheta = np.cos(angles).mean()
        yTheta = np.sin(angles).mean()

        self.theta = np.arctan2(yTheta, xTheta)

        self.x = selectedX - self.camera_offset * np.cos(self.theta)
        self.y = selectedY - self.camera_offset * np.sin(self.theta)

        return image
        
    def visualizeMarkers(self, image, corners, ids):
        ids = ids.flatten()
        # loop over the detected ArUCo corners  
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
                       
            # draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            cv2.putText(image, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        return image
        
    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            color = (0, 0, 255)

            if self.latest_image is not None and self.K is not None:
                image = self._bridge.compressed_imgmsg_to_cv2(self.latest_image)

                if image is None:
                    rospy.logwarn("Failed to convert image from ROS message!")
                    continue    

                corners, ids, _ = self.detectMarkers(image)

                if ids is not None:
                    self.localizationState = True
                    self.last_tag_time = rospy.Time.now()

                    image = self.visualizeMarkers(image, corners, ids)
                    image = self.getPose(image, corners, ids)
                    color = (0, 255, 0)

                    # Recalculate motion immediately from updated pose
                    self.navigate_to_current_target()

                else:
                    self.localizationState = False
                    if (self._ticks_left is not None and self._prev_ticks_left is not None and
                        self._ticks_right is not None and self._prev_ticks_right is not None):

                        delta_left = self._ticks_left - self._prev_ticks_left
                        delta_right = self._ticks_right - self._prev_ticks_right

                        R = 0.0318  # wheel radius [m]
                        L = 0.1     # wheelbase [m]
                        N = 135     # encoder resolution [ticks]

                        dL = 2 * np.pi * R * (delta_left / N)
                        dR = 2 * np.pi * R * (delta_right / N)

                        if(abs(dL - dR) <= 1e-6):
                            self.x = self.x + dL * np.cos(self.theta)
                            self.y = self.y + dL * np.sin(self.theta)
                        else:   
                            wd = (dR - dL)/L
                            RR = L * (dL + dR) / (2 * (dR - dL))

                            self.x = self.x + RR * np.sin(wd + self.theta) - RR * np.sin(self.theta)
                            self.y = self.y - RR * np.cos(wd + self.theta) + RR * np.cos(self.theta)
                            self.theta = self.normalize(self.theta + wd)
                        
                        color = (0, 0, 255)

                    # Assignment says stop or slow down and reacquire
                    dt = (rospy.Time.now() - self.last_tag_time).to_sec()

                    if dt > self.tag_timeout:
                        elapsed = (rospy.Time.now() - self.last_tag_time).to_sec()

                        cycle_time = self.search_rotate_seconds + self.search_stop_seconds
                        phase_time = elapsed % cycle_time

                        if phase_time < self.search_rotate_seconds:
                            self.publish_cmd(0.0, self.search_w)
                        else:
                            self.stop_robot()
                    else:
                        # optional small search rotation before full stop
                        self.navigate_to_current_target()

                # if ids is not None:
                #     self.localizationState = True
                #     image = self.visualizeMarkers(image, corners, ids)
                #     image = self.getPose(image, corners, ids)   
                #     color = (0, 255, 0)
                        
                # else:
                #     self.localizationState = False
                #     if (self._ticks_left is not None and self._prev_ticks_left is not None and
                #         self._ticks_right is not None and self._prev_ticks_right is not None):

                #         delta_left = self._ticks_left - self._prev_ticks_left
                #         delta_right = self._ticks_right - self._prev_ticks_right

                #         print("DELTA LEFT: ", delta_left)
                #         print("TICKS LEFT: ", self._ticks_left)
                #         print("PREV TICKS LEFT: ", self._prev_ticks_left)


                #         R = 0.0318  # wheel radius [m]
                #         L = 0.1     # wheelbase [m]
                #         N = 135     # encoder resolution [ticks]

                #         dL = 2 * np.pi * R * (delta_left / N)
                #         dR = 2 * np.pi * R * (delta_right / N)

                #         print("DL: ", dL)

                #         if(abs(dL - dR) <= 1e-6):
                #             print("IAM HERE")
                #             self.x = self.x + dL * np.cos(self.theta)
                #             self.y = self.y + dL * np.sin(self.theta)
                #         else:   
                #             wd = (dR - dL)/L
                #             RR = L * (dL + dR) / (2 * (dR - dL))

                #             self.x = self.x + RR * np.sin(wd + self.theta) - RR * np.sin(self.theta)
                #             self.y = self.y - RR * np.cos(wd + self.theta) + RR * np.cos(self.theta)
                #             self.theta = self.normalize(self.theta + wd)
                        
                #         color = (0, 0, 255)


                if self._ticks_left is not None:
                    self._prev_ticks_left = self._ticks_left
                if self._ticks_right is not None:
                    self._prev_ticks_right = self._ticks_right

                # draw odometry on screen
                cv2.putText(image, f"x={self.x * 100:.2f} cm", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image, f"y={self.y * 100:.2f} cm", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image, f"theta={self.theta:.2f} rad", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                self.draw_on_map()
                cv2.imshow(self._window, image)
                cv2.waitKey(1)

            rate.sleep()

    def reset_state(self):
        rospy.loginfo("Resetting all state variables...")
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._ticks_left = None
        self._ticks_right = None
        self._prev_ticks_left = None
        self._prev_ticks_right = None
        self.tagMap = {}
        self.localizationState = False
        self.latest_image = None

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.reset_state()
        cv2.destroyAllWindows()

if __name__ == '__main__':
   node = CameraReaderNode(node_name='camera_reader_node')
   node.run()

"""
RUN CODE:

dts devel run -R chicken -L localizer_package -X 

"""