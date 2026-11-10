"""ros2 launch ucr_dv_ros stack.launch.py profile:=baseline [ucr_dv_root:=/path/to/ucr-dv-stack]

Expects the simulator (fsds_lite --no-rpc or FSDS + fsds_ros2_bridge) to be publishing.
Export the profile first: `./ucr-dv export --profile baseline` (writes config/<profile>.json/.yaml).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _node(context):
    profile = LaunchConfiguration('profile').perform(context)
    root = LaunchConfiguration('ucr_dv_root').perform(context)
    share = get_package_share_directory('ucr_dv_ros')
    yaml = os.path.join(share, 'config', profile + '.yaml')
    params = [yaml] if os.path.isfile(yaml) else []
    params.append({'profile': profile})
    env = {'UCR_DV_ROOT': root} if root else {}
    return [Node(package='ucr_dv_ros', executable='stack_node', name='ucr_dv_stack', output='screen',
                 parameters=params, additional_env=env)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='baseline'),
        DeclareLaunchArgument('ucr_dv_root', default_value=os.environ.get('UCR_DV_ROOT', '')),
        OpaqueFunction(function=_node),
    ])
