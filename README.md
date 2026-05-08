# 🤖 Autonomous Robotics Projects

This repository contains a collection of robotics projects focused on autonomous navigation, localization, and probabilistic state estimation using ROS, computer vision, and differential-drive robotics.

The projects were developed on Duckietown-based robots and explore how perception, planning, and motion estimation interact in real robotic systems operating under noisy and incomplete observations.

---

# 📚 Projects

- [🧭 A* Autonomous Navigation](#-a-autonomous-navigation)
- [📍 Visual Localization with ArUco and Odometry Fusion](#-visual-localization-with-aruco-and-odometry-fusion)
- [🎯 Monte Carlo Localization](#-monte-carlo-localization)

---

# 🧭 A* Autonomous Navigation

![til](Visuals/AStarDemo.gif)

This project implements a complete autonomous navigation pipeline for a differential-drive robot using graph-based path planning and real-time localization.

A map of the environment is given and modeled as a weighted graph where intersections are represented as nodes and traversable connections are represented as edges with associated traversal costs. The robot computes the shortest path between a start node and a goal node using the A* search algorithm, then continuously tracks and follows the generated trajectory in real time.

Localization is performed using ArUco marker pose estimation combined with wheel encoder odometry. Whenever visual landmarks are visible, the robot estimates its global pose using camera observations and coordinate frame transformations. When visual feedback is temporarily lost, the system falls back to odometry-based motion prediction until visual localization becomes available again. If the robot loses the tags for a long time, meaning odometry drift is continuously increasing, the system temporarily stops navigation and actively searches for nearby landmarks by rotating in place until localization is recovered.

The navigation controller continuously rotates the robot toward the next target waypoint, aligns its heading, and applies differential-drive velocity commands to move toward the goal while correcting orientation errors in real time.

The system also includes a live top-down visualization interface displaying:
- the robot pose
- heading direction
- planned path
- graph nodes
- localization state
- map structure

🎥 Full Demonstration Video:
https://www.youtube.com/watch?v=maETFS25uTY

[View Project Folder](./A_Star_Navigation)

---

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

[View Project Folder](./VIsual_Localization)

---

# 🎯 Monte Carlo Localization

This project implements Monte Carlo Localization (MCL) using a particle filter to estimate robot pose in ambiguous environments containing repeated visual landmarks.

Instead of maintaining a single pose estimate, the system represents the robot belief as a distribution of particles spread across the environment. Each particle represents a possible robot pose hypothesis.

As the robot moves, wheel encoder odometry propagates the particles according to the differential-drive motion model. Visual observations from ArUco markers are then used to evaluate how likely each particle is to represent the true robot pose.

Particles with observations that better match the real camera measurements receive higher weights, while unlikely hypotheses gradually disappear through resampling. Additional Gaussian noise is injected during resampling to preserve state diversity and avoid particle collapse.

The project includes:
- motion prediction
- observation likelihood estimation
- particle weighting
- effective sample size evaluation
- probabilistic resampling
- real-time particle visualization

The resulting system demonstrates how probabilistic robotics techniques can localize a robot even under uncertainty, noisy measurements, and ambiguous landmark configurations.

[View Project Folder](./Monte_Carlo_Localization)

---
