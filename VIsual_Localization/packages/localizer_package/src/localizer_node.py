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
        
        self._window = "Localizer"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        
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

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.tagMap = {}
        self.localizationState = False

        self.K = None
        self.D = None

        rospy.on_shutdown(self.on_shutdown)

    def draw_on_map(self):
        self.map_canvas = self.base_map.copy()

        px = int(500 - self.y * 100)
        py = int(500 - self.x * 100)

        if 0 <= px < 1000 and 0 <= py < 1000:
            color = (0, 255, 0) if self.localizationState else (0, 0, 255)

            # Draw robot position
            cv2.circle(self.map_canvas, (px, py), 5, color, -1)

            # Arrow
            arrow_len = 30
            dx = -np.sin(self.theta)
            dy = -np.cos(self.theta)

            end_x = int(px + arrow_len * dx)
            end_y = int(py + arrow_len * dy)

            cv2.arrowedLine(self.map_canvas, (px, py), (end_x, end_y), color, 2, tipLength=0.3)

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

    def normalize(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def detectMarkers(self, image):
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        return corners, ids, rejected

    def getPose(self, image, corners, ids):
        cameraMatrix, distCoeffs = self.K, self.D
        markerLength = 0.065

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, markerLength, cameraMatrix, distCoeffs
        )

        xSum, ySum, angles = 0, 0, []

        for i in range(len(ids)):
            rvec, tvec = rvecs[i], tvecs[i]
            tag_id = ids[i][0]

            cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            R, _ = cv2.Rodrigues(rvec)
            yaw = np.arctan2(R[1,0], R[0,0])


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

            xSum += self.tagMap[tag_id][0] - world_dx
            ySum += self.tagMap[tag_id][1] - world_dy
            
            estimatedTheta = self.normalize(self.tagMap[tag_id][2] - yaw)
            angles.append(estimatedTheta)
        self.x = xSum/len(ids)
        self.y = ySum/len(ids)

        xTheta = np.cos(angles).mean()
        yTheta = np.sin(angles).mean()

        self.theta = np.arctan2(yTheta, xTheta)

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
                    image = self.visualizeMarkers(image, corners, ids)
                    image = self.getPose(image, corners, ids)
                    color = (0, 255, 0)
                        
                else:
                    self.localizationState = False
                    if (self._ticks_left is not None and self._prev_ticks_left is not None and
                        self._ticks_right is not None and self._prev_ticks_right is not None):

                        delta_left = self._ticks_left - self._prev_ticks_left
                        delta_right = self._ticks_right - self._prev_ticks_right

                        print("DELTA LEFT: ", delta_left)
                        print("TICKS LEFT: ", self._ticks_left)
                        print("PREV TICKS LEFT: ", self._prev_ticks_left)


                        R = 0.0318  # wheel radius [m]
                        L = 0.1     # wheelbase [m]
                        N = 135     # encoder resolution [ticks]

                        dL = 2 * np.pi * R * (delta_left / N)
                        dR = 2 * np.pi * R * (delta_right / N)

                        print("DL: ", dL)

                        if(abs(dL - dR) <= 1e-6):
                            print("IAM HERE")
                            self.x = self.x + dL * np.cos(self.theta)
                            self.y = self.y + dL * np.sin(self.theta)
                        else:   
                            wd = (dR - dL)/L
                            RR = L * (dL + dR) / (2 * (dR - dL))

                            self.x = self.x + RR * np.sin(wd + self.theta) - RR * np.sin(self.theta)
                            self.y = self.y - RR * np.cos(wd + self.theta) + RR * np.cos(self.theta)
                            self.theta = self.normalize(self.theta + wd)
                        
                        color = (0, 0, 255)


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