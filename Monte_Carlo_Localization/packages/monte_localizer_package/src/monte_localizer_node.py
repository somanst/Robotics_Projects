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

R = 0.0318  # wheel radius [m]
L = 0.1     # wheelbase [m]
N = 135     # encoder resolution [ticks]


# ---------------------------------------------------------------------
# Robot starting pose in world coordinates.
# Set this to where the robot is physically placed when you start the
# node. The odometry-only baseline starts here and integrates wheel
# encoders from this pose. The particle filter does NOT use this; it
# spawns particles uniformly across the map so it can still localize
# from scratch.
#
# Current default: near the NE corner of the room, facing outward
# (toward the NE corner) so the robot does not see any tag at startup
# and the particle cloud stays uniformly random until the robot is
# turned to look at tags.
# ---------------------------------------------------------------------
ROBOT_INITIAL_X = 0.85
ROBOT_INITIAL_Y = 0.85
ROBOT_INITIAL_THETA = np.pi / 4

# ---------------------------------------------------------------------
# How long (in seconds) trajectory lines stay visible on the top-down
# map before fading out. Older points are pruned every frame so the
# red (odometry) and green (filter estimate) trails do not pile up
# during long runs and the visualization stays readable.
# ---------------------------------------------------------------------
TRAJECTORY_TRAIL_SECONDS = 15.0


class CameraReaderNode(DTROS):
    def __init__(self, node_name):

        super(CameraReaderNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.VISUALIZATION
        )

        self.latest_image = None
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()

        self._window = "Monte Carlo Localizer"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        self.map_w = 1000
        self.map_h = 1000

        # pixels per meter
        self.scale = 800

        # screen margin
        self.margin = 100

        # world origin on screen
        # Bottom-left corner of the lab room maps to (margin, height - margin)
        # in pixel coordinates. With scale=800, a 0.90 m room occupies 720 px
        # and stays inside the 1000x1000 canvas with room to spare for labels.
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

        # Lab tag map.
        # Origin (0, 0) is at the bottom-left corner of the lab room.
        # Room is approximately 0.90 m x 0.90 m.
        # Tag theta = direction the tag's front face points, i.e. the normal
        # vector pointing AWAY from the wall INTO the room. The filter uses
        # this to predict the relative yaw a robot would observe for the tag.
        self.tagMap = {
            # ----- South wall (y = 0.00), tags face +y -----
            0: np.array([0.33, 0.00,  np.pi / 2]),
            1: np.array([0.50, 0.00,  np.pi / 2]),

            # ----- East wall (x = 0.90), tags face -x -----
            2: np.array([0.90, 0.25,  np.pi]),
            3: np.array([0.90, 0.45,  np.pi]),

            # ----- West wall (x = 0.00), tags face +x -----
            4: np.array([0.00, 0.42,  0.0]),
            5: np.array([0.00, 0.71,  0.0]),

            # ----- North wall (y = 0.90), tags face -y -----
            6: np.array([0.05, 0.90, -np.pi / 2]),
            7: np.array([0.55, 0.90, -np.pi / 2]),
        }
        self.localizationState = False
        self.draw_fixed_tags()

        self.K = None
        self.D = None

        rospy.on_shutdown(self.on_shutdown)
        self.frameCount = 0

        self.particles = []
        self.weights = []
        self.particleCount = 500

        # Odometry-only pose (no filter correction).
        # This pose is updated only from wheel encoders and never
        # corrected by sensor observations. It is the baseline against
        # which we compare the particle filter's estimate.
        # Seeded from the manual ROBOT_INITIAL_* constants at the top
        # of this file so the red trajectory starts at the robot's real
        # physical placement.
        self.odom_x = ROBOT_INITIAL_X
        self.odom_y = ROBOT_INITIAL_Y
        self.odom_theta = ROBOT_INITIAL_THETA

        # Trajectory histories for visualization.
        # odom_trajectory:     raw odometry path (drifts over time)
        # estimate_trajectory: particle filter weighted-mean path
        self.odom_trajectory = []
        self.estimate_trajectory = []

        self.initializeParticles()

        self.draw_on_map()

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
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        arucoParams = cv2.aruco.DetectorParameters_create()

        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeMax = 35
        arucoParams.minMarkerPerimeterRate = 0.03

        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        return corners, ids, rejected

    def camera_to_robot_plane(self, tvec, Rmat):
        # OpenCV camera frame:
        # x = right, y = down, z = forward
        #
        # Robot/world ground frame:
        # x = forward, y = left, z = up

        R_robot_cam = np.array([
            [0,  0,  1],   # robot x  = camera z
            [-1, 0,  0],   # robot y  = -camera x
            [0, -1,  0]    # robot z  = -camera y
        ])

        t_cam = tvec.reshape(3, 1)
        t_robot = R_robot_cam @ t_cam

        local_x = t_robot[0, 0]   # forward on movement plane
        local_y = t_robot[1, 0]   # left/right on movement plane

        # Marker normal in camera frame
        marker_normal_cam = Rmat @ np.array([[0], [0], [1]])

        # Marker normal in robot frame
        marker_normal_robot = R_robot_cam @ marker_normal_cam

        # Project normal onto ground plane
        nx = marker_normal_robot[0, 0]
        ny = marker_normal_robot[1, 0]

        observed_yaw = self.normalize(np.arctan2(ny, nx))

        return local_x, local_y, observed_yaw

    def get_observed_tags(self, image, corners, ids):
        cameraMatrix, distCoeffs = self.K, self.D
        markerLength = 0.065

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            markerLength,
            cameraMatrix,
            distCoeffs
        )

        observations = []

        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]

            cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            Rmat, _ = cv2.Rodrigues(rvec)

            local_x, local_y, observed_yaw = self.camera_to_robot_plane(tvec, Rmat)

            distance = np.sqrt(local_x**2 + local_y**2)
            bearing = np.arctan2(local_y, local_x)

            observations.append([distance, bearing, observed_yaw])

            print(f"LOCAL_X:{local_x}, LOCAL_Y{local_y}")
            print("OBSERVATIONS:", observations)

        return image, observations

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

    def updateOdometry(self, x, y, theta, dL, dR):
        if(abs(dL - dR) <= 1e-6):
            x = x + dL * np.cos(theta)
            y = y + dL * np.sin(theta)
        else:
            wd = (dR - dL)/L
            RR = L * (dL + dR) / (2 * (dR - dL))

            dx = RR * np.sin(theta + wd) - RR * np.sin(theta)
            dy = -RR * np.cos(theta + wd) + RR * np.cos(theta)

            x = x + dx
            y = y + dy
            theta = self.normalize(theta + wd)

        return x, y, theta

    def updateAllParticlesOdometry(self, dL, dR):
        # Motion model noise parameters used in the prediction step.
        # alpha scales noise with motion magnitude (drift error grows
        # with traveled distance), beta is a small baseline noise that
        # is always present even when the robot is essentially still.
        # This is the standard odometry noise model in probabilistic
        # robotics.
        alpha = 0.05    # 5% of traveled distance
        beta = 0.002    # 2 mm constant component

        for i in range(len(self.particles)):
            particle = self.particles[i]
            particleX, particleY, particleTheta = particle[0], particle[1], particle[2]

            # Independent Gaussian noise on each wheel displacement.
            sigma_L = alpha * abs(dL) + beta
            sigma_R = alpha * abs(dR) + beta

            noisy_dL = dL + np.random.normal(0, sigma_L)
            noisy_dR = dR + np.random.normal(0, sigma_R)

            x, y, theta = self.updateOdometry(
                particleX,
                particleY,
                particleTheta,
                noisy_dL,
                noisy_dR
            )

            self.particles[i] = [x, y, theta]

    def predict_observations_for_particle(self, particle, max_range=1.5, fov=np.pi/2):
        x, y, theta = particle

        # convert robot center → camera position
        cam_x = x + 0.15 * np.cos(theta)
        cam_y = y + 0.15 * np.sin(theta)

        predicted = []

        for tag_id, tag_pose in self.tagMap.items():
            tag_x, tag_y, tag_theta = tag_pose

            dx = tag_x - cam_x
            dy = tag_y - cam_y

            distance = np.sqrt(dx*dx + dy*dy)
            bearing = self.normalize(np.arctan2(dy, dx) - theta)

            predicted_yaw = self.normalize(tag_theta - theta)

            if distance <= max_range and abs(bearing) <= fov / 2:
                predicted.append([distance, bearing, predicted_yaw])

        return predicted

    def observation_likelihood(self, real_observations, predicted_observations):
        if len(real_observations) == 0:
            return 1.0

        if len(predicted_observations) == 0:
            return 1e-9

        total_weight = 1.0

        sigma_dist = 0.1
        sigma_bearing = 0.2
        sigma_yaw = 0.2

        for real_dist, real_bearing, real_yaw in real_observations:
            best_prob = 0.0

            for pred_dist, pred_bearing, pred_yaw in predicted_observations:
                dist_error = real_dist - pred_dist
                bearing_error = self.normalize(real_bearing - pred_bearing)
                yaw_error = self.normalize(real_yaw - pred_yaw)

                p_dist = np.exp(-0.5 * (dist_error / sigma_dist) ** 2)
                p_bearing = np.exp(-0.5 * (bearing_error / sigma_bearing) ** 2)
                p_yaw = np.exp(-0.5 * (yaw_error / sigma_yaw) ** 2)

                prob = p_dist * p_bearing * p_yaw
                best_prob = max(best_prob, prob)

            total_weight *= max(best_prob, 1e-9)

        return total_weight

    def update_particle_weights(self, real_observations):
        weights = []

        for particle in self.particles:
            predicted = self.predict_observations_for_particle(particle)
            w = self.observation_likelihood(real_observations, predicted)
            weights.append(w)

        weights = np.array(weights)

        if np.sum(weights) < 1e-12:
            self.weights = np.ones(self.particleCount) / self.particleCount
        else:
            self.weights = weights / np.sum(weights)

    def resample_with_noise(self, particles, weights):
        weights = weights / np.sum(weights)

        indices = np.random.choice(
            np.arange(len(particles)),
            size=len(particles),
            replace=True,
            p=weights
        )

        new_particles = particles[indices].copy()

        noise = np.random.normal(
            0,
            [0.02, 0.02, 0.05],
            size=new_particles.shape
        )

        new_particles += noise
        new_particles[:, 2] = [self.normalize(t) for t in new_particles[:, 2]]

        return new_particles

    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights ** 2)

    def initializeParticles(self):
        particles = []

        # Get map bounds from tag positions
        tag_positions = np.array([
            [pose[0], pose[1]]
            for pose in self.tagMap.values()
        ])

        min_x = np.min(tag_positions[:, 0])
        max_x = np.max(tag_positions[:, 0])
        min_y = np.min(tag_positions[:, 1])
        max_y = np.max(tag_positions[:, 1])

        # Optional padding so particles can spawn around tags too
        padding = 0.5  # meters

        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        for _ in range(self.particleCount):
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            theta = np.random.uniform(-np.pi, np.pi)

            particles.append([x, y, theta])

        self.particles = np.array(particles)

        # Equal probability at start
        self.weights = np.ones(self.particleCount) / self.particleCount

        rospy.loginfo(f"Initialized {self.particleCount} free-space particles")

    def camera_to_robot_center(self, x, y, theta):
        offset = 0.1  # meters

        robot_x = x - offset * np.cos(theta)
        robot_y = y - offset * np.sin(theta)

        return robot_x, robot_y, theta

    def draw_particles(self, canvas):
        if self.particles is None or len(self.particles) == 0:
            return

        arrow_len_m = 0.03

        # Normalize weights for color mapping. The brightest yellow
        # corresponds to the heaviest particle; lower-weight particles
        # fade toward darker shades so the weight distribution is visible
        # at a glance.
        if self.weights is not None and len(self.weights) == len(self.particles):
            max_w = np.max(self.weights)
            if max_w < 1e-12:
                normalized = np.zeros_like(self.weights)
            else:
                normalized = self.weights / max_w
            best_idx = int(np.argmax(self.weights))
        else:
            normalized = np.ones(len(self.particles)) / len(self.particles)
            best_idx = None

        for i, (x, y, theta) in enumerate(self.particles):

            px, py = self.world_to_pixel(x, y)

            end_x = x + arrow_len_m * np.cos(theta)
            end_y = y + arrow_len_m * np.sin(theta)

            ex, ey = self.world_to_pixel(end_x, end_y)

            if 0 <= px < self.map_w and 0 <= py < self.map_h:

                if i == best_idx:
                    # Most probable particle: solid red, larger circle and arrow.
                    cv2.circle(canvas, (px, py), 7, (0, 0, 255), -1)
                    cv2.arrowedLine(canvas, (px, py), (ex, ey), (0, 0, 255), 2, tipLength=0.4)
                else:
                    # Normal particles: yellow circle + heading arrow.
                    # Brightness is scaled by the particle's weight so that
                    # high-weight clusters appear vivid while low-weight
                    # outliers fade. The arrow is preserved so the pose
                    # (heading) of every particle remains visible.
                    w = float(normalized[i])
                    color = (
                        int(40 + 60 * w),       # B: slight blue tint at low weight
                        int(60 + 195 * w),      # G
                        int(60 + 195 * w)       # R  -> yellow when bright
                    )
                    cv2.circle(canvas, (px, py), 2, color, -1)
                    cv2.arrowedLine(canvas, (px, py), (ex, ey), color, 1, tipLength=0.4)

    def draw_on_map(self):
        self.map_canvas = self.base_map.copy()

        # Drop trajectory points older than TRAJECTORY_TRAIL_SECONDS so
        # the rendered trails do not accumulate forever. The lists store
        # tuples of (x, y, timestamp_seconds).
        now = rospy.Time.now().to_sec()
        cutoff = now - TRAJECTORY_TRAIL_SECONDS

        # Prune in place (lists stay chronological because we only
        # append in real time).
        self.odom_trajectory = [
            p for p in self.odom_trajectory if p[2] >= cutoff
        ]
        self.estimate_trajectory = [
            p for p in self.estimate_trajectory if p[2] >= cutoff
        ]

        # Draw odometry-only trajectory in RED.
        # This is the path the robot would think it took if it trusted
        # the wheel encoders alone, with no sensor-based correction.
        # It will drift away from the true path over time.
        if len(self.odom_trajectory) > 1:
            for i in range(1, len(self.odom_trajectory)):
                x1, y1, _ = self.odom_trajectory[i - 1]
                x2, y2, _ = self.odom_trajectory[i]
                p1 = self.world_to_pixel(x1, y1)
                p2 = self.world_to_pixel(x2, y2)
                cv2.line(self.map_canvas, p1, p2, (0, 0, 255), 2)

        # Draw particle-filter estimate trajectory in GREEN.
        # This is the weighted mean over all particles, which is the
        # canonical filter pose estimate as specified by the project.
        # It should track the true robot pose once the filter
        # converges to a single cluster.
        if len(self.estimate_trajectory) > 1:
            for i in range(1, len(self.estimate_trajectory)):
                x1, y1, _ = self.estimate_trajectory[i - 1]
                x2, y2, _ = self.estimate_trajectory[i]
                p1 = self.world_to_pixel(x1, y1)
                p2 = self.world_to_pixel(x2, y2)
                cv2.line(self.map_canvas, p1, p2, (0, 255, 0), 2)

        self.draw_particles(self.map_canvas)

        # Legend so the viewer can tell which line is which.
        cv2.putText(self.map_canvas, "RED   = odometry only",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        cv2.putText(self.map_canvas, "GREEN = filter estimate (weighted mean)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.imshow("Top-Down Map", self.map_canvas)

    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            self.frameCount += 1
            if self.latest_image is not None and self.K is not None:
                image = self._bridge.compressed_imgmsg_to_cv2(self.latest_image)

                if image is None:
                    rospy.logwarn("Failed to convert image from ROS message!")
                    continue

                if (self._ticks_left is not None and self._prev_ticks_left is not None and
                    self._ticks_right is not None and self._prev_ticks_right is not None):

                    delta_left = self._ticks_left - self._prev_ticks_left
                    delta_right = self._ticks_right - self._prev_ticks_right

                    dL = 2 * np.pi * R * (delta_left / N)
                    dR = 2 * np.pi * R * (delta_right / N)

                    self.updateAllParticlesOdometry(dL, dR)

                    # Update the naive odometry-only pose. This pose is
                    # NEVER corrected by sensor observations; it is the
                    # baseline we plot in red to show what the robot would
                    # believe its location was if it trusted the wheel
                    # encoders alone. Over time it drifts away from the
                    # true path.
                    self.odom_x, self.odom_y, self.odom_theta = self.updateOdometry(
                        self.odom_x, self.odom_y, self.odom_theta, dL, dR
                    )
                    # Store the point together with its timestamp so the
                    # renderer can drop entries older than the trail
                    # duration.
                    self.odom_trajectory.append(
                        (self.odom_x, self.odom_y, rospy.Time.now().to_sec())
                    )

                corners, ids, _ = self.detectMarkers(image)

                if ids is not None:
                    self.localizationState = True
                    self.last_tag_time = rospy.Time.now()

                    image = self.visualizeMarkers(image, corners, ids)
                    image, observations = self.get_observed_tags(image, corners, ids)

                    # self.update_particle_weights(observations)
                    # self.particles = self.resample_with_noise(self.particles, self.weights)
                    # self.weights = np.ones(self.particleCount) / self.particleCount

                    self.update_particle_weights(observations)

                    # Compute the filter's pose estimate as the WEIGHTED
                    # MEAN of all particles. The project specifies this
                    # as the canonical filter pose estimate.
                    #
                    # For the orientation we use the circular mean
                    # (atan2 of weighted sin/cos), because plain
                    # averaging breaks at the +/- pi wrap-around.
                    #
                    # Note: while the filter still has multiple plausible
                    # clusters, the weighted mean can sit between them
                    # and look meaningless. This is expected behaviour
                    # and disappears once the filter collapses to a
                    # single cluster around the true pose.
                    est_x = np.sum(self.weights * self.particles[:, 0])
                    est_y = np.sum(self.weights * self.particles[:, 1])
                    est_cos = np.sum(self.weights * np.cos(self.particles[:, 2]))
                    est_sin = np.sum(self.weights * np.sin(self.particles[:, 2]))
                    est_theta = np.arctan2(est_sin, est_cos)
                    # Store the point with its timestamp so old segments
                    # fade out of the rendered trail.
                    self.estimate_trajectory.append(
                        (est_x, est_y, rospy.Time.now().to_sec())
                    )

                    best_idx = np.argmax(self.weights)
                    best_particle = self.particles[best_idx]
                    best_weight = self.weights[best_idx]
                    predicted = self.predict_observations_for_particle(best_particle)

                    print("\n===== BEST PARTICLE =====")
                    print(f"Index: {best_idx}")
                    print(f"Pose: x={best_particle[0]:.3f}, y={best_particle[1]:.3f}, theta={best_particle[2]:.3f}")
                    print(f"Weight: {best_weight:.8f}")

                    print("Predicted observations:")
                    for j, (dist, bearing, yaw) in enumerate(predicted):
                        print(
                            f"  pred {j}: "
                            f"dist={dist:.3f}, "
                            f"bearing={bearing:.3f}, "
                            f"yaw={yaw:.3f}"
                        )

                    print("=========================\n")

                    ess = self.effective_sample_size()

                    if ess < 0.5 * self.particleCount and self.frameCount % 4 == 0:
                        self.particles = self.resample_with_noise(self.particles, self.weights)
                        self.weights = np.ones(self.particleCount) / self.particleCount


                if self._ticks_left is not None:
                    self._prev_ticks_left = self._ticks_left
                if self._ticks_right is not None:
                    self._prev_ticks_right = self._ticks_right

                self.draw_on_map()
                cv2.imshow(self._window, image)
                cv2.waitKey(1)

            rate.sleep()

    def reset_state(self):
        rospy.loginfo("Resetting all state variables...")
        self._ticks_left = None
        self._ticks_right = None
        self._prev_ticks_left = None
        self._prev_ticks_right = None
        self.latest_image = None

        # Reset the odometry-only baseline back to the manual starting
        # pose, and clear the visualization buffers so a fresh run does
        # not inherit stale paths.
        self.odom_x = ROBOT_INITIAL_X
        self.odom_y = ROBOT_INITIAL_Y
        self.odom_theta = ROBOT_INITIAL_THETA
        self.odom_trajectory = []
        self.estimate_trajectory = []

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.reset_state()
        cv2.destroyAllWindows()

if __name__ == '__main__':
   node = CameraReaderNode(node_name='camera_reader_node')
   node.run()



"""
RUN CODE:

dts devel run -R hostname -L monte_localizer_package -X 

dts duckiebot keyboard_control hostname
"""