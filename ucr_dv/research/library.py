"""
The fsd-research reading library, indexed: which paper backs which algorithm.

Files live in ../fsd-research (paths.RESEARCH); nothing is copied. The stage
tags let the Research page show "papers for this stage" next to the
algorithms that implement them.
"""
from __future__ import annotations

import os

from .. import paths

PAPERS = [
    {'file': 'papers/2019_AMZ_driverless_full_system_1905.05150.pdf', 'title': 'AMZ Driverless: The Full Autonomous Racing System',
     'team': 'ETH Zurich (AMZ)', 'year': 2019, 'stages': ['lidar', 'camera', 'fusion', 'odometry', 'mapping', 'planner', 'lateral'],
     'key': 'Whole-car reference: camera+LiDAR pipelines, velocity EKF, FastSLAM 2.0, Delaunay path search, MPC. '
            'Read first for the big picture.'},
    {'file': 'papers/2020_AMZ_accurate_mapping_planning_2003.05266.pdf', 'title': 'Accurate Mapping and Planning for Autonomous Racing',
     'team': 'ETH Zurich (AMZ)', 'year': 2020, 'stages': ['fusion', 'mapping', 'planner'],
     'key': 'Early camera-LiDAR fusion, two-layer map, boundary estimation: unknown-track speed 3 -> 12 m/s.'},
    {'file': 'papers/2020_ETH_time_optimal_trajectory_tracking_2003.04882.pdf', 'title': 'Time-Optimal Trajectory Tracking (NMPC)',
     'team': 'ETH Zurich', 'year': 2020, 'stages': ['speed', 'lateral'],
     'key': 'Offline racing line + NMPC tracking: ~20 % lap time gain. Target for trackdrive (known map).'},
    {'file': 'papers/2020_FSOCO_cone_dataset_2012.07139.pdf', 'title': 'FSOCO: The Formula Student Objects in Context Dataset',
     'team': 'FS community', 'year': 2020, 'stages': ['camera'],
     'key': '11.5k images / 220k labelled cones. Where to get training data for a learned camera detector.'},
    {'file': 'papers/2022_KIT_software_stack_that_won_FSD_2210.10933.pdf', 'title': 'The Software Stack that Won FSD 2021',
     'team': 'KIT (KA-RaceIng)', 'year': 2022, 'stages': ['lidar', 'mapping', 'planner', 'lateral', 'longitudinal'],
     'key': 'LiDAR-only (cameras left off), GraphSLAM beat EKF-SLAM, triangulation + spline, MPC + PI speed.'},
    {'file': 'papers/2024_IIT_Bombay_driverless_stack_2408.06113.pdf', 'title': 'IIT Bombay Racing Driverless Stack',
     'team': 'IIT Bombay', 'year': 2024, 'stages': ['camera', 'lidar', 'fusion', 'mapping', 'planner', 'lateral'],
     'key': 'YOLOv5, three ranging methods, EKF-SLAM, Pure Pursuit beat Stanley in FSDS, skidpad template.'},
    {'file': 'papers/2025_midpoint_pure_pursuit_pipeline_2511.11310.pdf', 'title': 'Midpoint + Pure Pursuit pipeline (CARLA)',
     'team': 'Heriot-Watt Dubai', 'year': 2025, 'stages': ['camera', 'lidar', 'fusion', 'planner', 'lateral'],
     'key': 'YOLOv8 + DBSCAN (eps 0.5, min 5), confidence-weighted fusion, adaptive pure pursuit. Best first read.'},
    {'file': 'papers/2025_YOLOv11_spline_racing_line_2512.03886.pdf', 'title': 'YOLOv11 + spline racing line',
     'team': 'Fontan-Costas et al.', 'year': 2025, 'stages': ['camera', 'odometry', 'planner', 'lateral'],
     'key': 'Camera-only (ZED 2i), RTK-GNSS filter, spline racing line on a Jetson Orin NX with ROS 2.'},
    {'file': 'papers/thesis_Monash_Slomoi_path_planning_control.pdf', 'title': 'Monash thesis: path planning and control',
     'team': 'Monash', 'year': 2018, 'stages': ['planner', 'lateral'],
     'key': 'Discovery lap with MPC, then racing-line optimisation.'},
    {'file': 'papers/thesis_Polito_global_mapping_SLAM.pdf', 'title': 'PoliTo thesis: global mapping / SLAM',
     'team': 'Politecnico di Torino', 'year': 2021, 'stages': ['mapping', 'odometry'],
     'key': 'Building the cone map with SLAM (ZED + Velodyne + SBG); Gazebo and RC-car tests.'},
]


def library():
    out = []
    for p in PAPERS:
        q = dict(p)
        q['path'] = os.path.join(paths.RESEARCH, p['file'])
        q['available'] = os.path.isfile(q['path'])
        out.append(q)
    return out


def index_html():
    path = os.path.join(paths.RESEARCH, 'index.html')
    return path if os.path.isfile(path) else None
