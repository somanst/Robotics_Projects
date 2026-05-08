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
        self.margin = 150

        # world origin on screen
        self.origin_px = (self.map_w / 2, self.map_h / 2)
        
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

        grid_step = 0.25

        self.tagMap = {
            #0:  np.array([0 * grid_step, 1 * grid_step, -np.pi / 2]),
            #1:  np.array([1 * grid_step, 0 * grid_step, 0.0]),
            2:  np.array([-1 * grid_step, 0 * grid_step, 0]),
            3:  np.array([0 * grid_step, -1 * grid_step, 0]),
            4:  np.array([0 * grid_step, 1 * grid_step, 0])
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

    # def draw_on_map(self, x, y, theta):
    #     self.map_canvas = self.base_map.copy()

    #     px, py = self.world_to_pixel(x, y)

    #     if 0 <= px < self.map_w and 0 <= py < self.map_h:
    #         color = (0, 255, 0) if self.localizationState else (0, 0, 255)

    #         cv2.circle(self.map_canvas, (px, py), 6, color, -1)

    #         arrow_len_m = 0.10  # 10 cm arrow in world space
    #         end_x = x + arrow_len_m * np.cos(theta)
    #         end_y = y + arrow_len_m * np.sin(theta)
    #         ex, ey = self.world_to_pixel(end_x, end_y)

    #         cv2.arrowedLine(self.map_canvas, (px, py), (ex, ey), color, 2, tipLength=0.3)

    #     cv2.imshow("Top-Down Map", self.map_canvas)

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

            # SAME yaw logic as your old getPose()
            observed_yaw = np.arctan2(Rmat[-1, 0], Rmat[0, 0])

            # SAME local position logic as your old getPose()
            local_x = tvec[0][2]/2
            local_y = -tvec[0][0]/2


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
        for i in range(len(self.particles)):
            particle = self.particles[i]
            particleX, particleY, particleTheta = particle[0], particle[1], particle[2]

            noisy_dL = dL #+ np.random.normal(0, 0.005)
            noisy_dR = dR #+ np.random.normal(0, 0.005)

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

        # get most probable particle index
        best_idx = None
        if self.weights is not None and len(self.weights) == len(self.particles):
            best_idx = np.argmax(self.weights)

        for i, (x, y, theta) in enumerate(self.particles):

            px, py = self.world_to_pixel(x, y)

            end_x = x + arrow_len_m * np.cos(theta)
            end_y = y + arrow_len_m * np.sin(theta)

            ex, ey = self.world_to_pixel(end_x, end_y)

            if 0 <= px < self.map_w and 0 <= py < self.map_h:

                if i == best_idx:
                    # most probable particle = red and bigger
                    cv2.circle(canvas, (px, py), 7, (0, 0, 255), -1)
                    cv2.arrowedLine(canvas, (px, py), (ex, ey), (0, 0, 255), 2, tipLength=0.4)
                else:
                    # normal particles = yellow
                    cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1)
                    cv2.arrowedLine(canvas, (px, py), (ex, ey), (0, 255, 255), 1, tipLength=0.4)

    def draw_on_map(self):
        self.map_canvas = self.base_map.copy()
        self.draw_particles(self.map_canvas)
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

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.reset_state()
        cv2.destroyAllWindows()

if __name__ == '__main__':
   node = CameraReaderNode(node_name='camera_reader_node')
   node.run()

"""
RUN CODE:

dts devel run -R chicken -L monte_localizer_package -X 

"""