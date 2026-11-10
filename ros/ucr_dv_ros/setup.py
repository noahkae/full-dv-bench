import glob
import os

from setuptools import setup

package_name = 'ucr_dv_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.json') + glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UCalgary Racing Driverless',
    maintainer_email='dv@ucalgaryracing.ca',
    description='ROS 2 wrapper for the ucr_dv driverless stack',
    license='MIT',
    entry_points={'console_scripts': ['stack_node = ucr_dv_ros.stack_node:main']},
)
