# 🧭 A* Autonomous Navigation

![til](../Visuals/AStarDemo.gif)

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
