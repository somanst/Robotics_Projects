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
