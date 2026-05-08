# 📍 Visual Localization with ArUco and Odometry Fusion

This project focuses on robot localization using visual landmark observations and wheel encoder odometry.

The system detects ArUco markers in real time using OpenCV and estimates their relative pose with respect to the onboard camera. These observations are transformed into the world frame to estimate the robot position and orientation within the environment.

To improve robustness, wheel odometry is fused with visual observations so the robot can continue estimating its motion even when landmarks are temporarily outside the camera field of view.

The project explores practical robotics challenges such as:
- noisy visual observations
- coordinate frame transformations
- orientation estimation
- intermittent sensing
- differential-drive motion estimation

A live visualization interface displays the robot trajectory and orientation on a dynamically updated top-down map.
