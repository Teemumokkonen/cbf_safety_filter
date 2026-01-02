# Control barrier function based safety filter

A lightweight Control Barrier Function (CBF)–based safety filter for obstacle avoidance in mobile robots.

This repository provides a simple, modular implementation of CBFs for enforcing safety constraints on unicycle‑type robots. The filter sits between your nominal controller and the robot, modifying velocity commands only when necessary to maintain safety.

The project currently includes two types of barrier constraints:

* Geometric circular obstacle constraint  
    * A basic analytic CBF for circular obstacles.

* Costmap‑based safety constraint  
    A distance‑transform‑driven CBF that works directly with Nav2 costmaps or any 2D occupancy grid. This allows obstacle avoidance in arbitrary environments without hand‑crafted geometry.

This project was developed primarily as a self‑educational exploration of control barrier functions and to explore more how to improve map-based obstacle avoidance.

# Note

The implementation has only been tested in simulation.
On a real robot, the filter may produce rapid or aggressive corrective motions unless tuned carefully.