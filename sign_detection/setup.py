import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sign_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'images'), glob('images/*.png')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
         (os.path.join('share', package_name, 'model_weights'), glob('model_weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='p1xta',
    maintainer_email='daria.petrova496@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detect_sign = sign_detection.detect_sign:main'
        ],
    },
)
