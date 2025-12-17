import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'construction_and_overtaking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AutoRace Team',
    maintainer_email='autorace@example.com',
    description='Construction site navigation and car overtaking module for AutoRace 2025',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_avoidance = construction_and_overtaking.obstacle_avoidance:main'
        ],
    },
)
