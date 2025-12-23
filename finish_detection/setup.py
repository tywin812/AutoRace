from setuptools import setup
import os
from glob import glob

package_name = 'finish_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AutoRace Team',
    maintainer_email='you@example.com',
    description='Finish line detection using gradient-based checkered pattern analysis',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'finish_detector = finish_detection.finish_detector:main',
        ],
    },
)
