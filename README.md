# Multi-Modal Path Planning on Constraint Manifolds for Robotic Manipulation

Master's thesis project on constraint-manifold motion planning for robotic manipulation under payload and torque limitations.

The planner combines supported sliding and free-space transport to generate **slide-lift-slide** trajectories. Environmental support is treated as an additional manipulation mode, allowing the robot to move payloads through configurations that may not be feasible during unsupported carrying.

## 4.1 kg Payload Demonstration

[![4.1 kg payload experiment](hardware_results/4.1kg/traj_all.png)](hardware_results/4.1kg/overlay_video.mp4)

**Click the image above to open the 4.1 kg hardware demonstration video.**

Additional experiment videos:

- [4.1 kg overlay video](hardware_results/4.1kg/overlay_video.mp4)
- [4.1 kg animation](hardware_results/4.1kg/animation_video.mp4)
- [2.1 kg overlay video](hardware_results/2.1kg/overlay_video.mp4)
- [2.1 kg animation](hardware_results/2.1kg/animation_video.mp4)

## Overview

The project investigates multi-modal robotic manipulation in which the robot can transition between:

- supported motion along a surface,
- free-space transport,
- and transitions between these constraint manifolds.

Torque feasibility is evaluated during planning. When unsupported carrying becomes infeasible under the selected torque limits, environmental support can be exploited to continue the transport task.

The repository contains the planner implementation, constraint handling, OMPL-based constrained planning, payload modelling, foliation-based free-carry refinement, experimental scripts, trajectory data and hardware-result visualisations.

## Repository Structure

- `payload_planner/` - main planning and analysis implementation
- `payload_planner/experiments/` - thesis experiment scripts and processed data
- `hardware_results/2.1kg/` - 2.1 kg hardware trajectories, plots and videos
- `hardware_results/4.1kg/` - 4.1 kg hardware trajectories, plots and videos
- `third_party/franka_description/` - licence and notice for redistributed Franka assets
- `docs/` - additional project documentation

## Main Components

- **`run_panda.py`** - main planning entry point
- **`transport.py`** - multi-modal transport sequence generation
- **`ompl_planner.py`** - constrained motion planning using OMPL
- **`constraints.py`** - constraint-manifold and mode definitions
- **`payload_model.py`** - payload modelling and configuration
- **`robot_model.py`** - robot kinematics and dynamics interface
- **`collision.py`** - environment and collision geometry
- **`transfer_foliation.py`** - refinement of the free-carry manifold
- **`experiments/`** - experimental scripts and processed thesis data

## Hardware Results

Hardware experiment data for the **2.1 kg** and **4.1 kg** payload cases are included in `hardware_results/`.

Each directory contains measured trajectory data, trajectory plots, visualisations and experiment videos.

## Dependencies

The planner is implemented in Python and uses libraries including:

- NumPy
- Pinocchio
- OMPL

Some visualisation and analysis scripts require additional packages. Detailed installation and reproduction instructions will be added separately.

## Thesis

**Multi-Modal Path Planning on Constraint Manifolds for Robotic Manipulation**

Master's thesis project investigating multi-modal constrained motion planning and the use of environmental support for robotic payload transport.

## Third-Party Assets

Robot mesh assets originating from `franka_description` are included where required by the URDF models used by the project.

The corresponding upstream licence and notice are preserved in:

`third_party/franka_description/`
