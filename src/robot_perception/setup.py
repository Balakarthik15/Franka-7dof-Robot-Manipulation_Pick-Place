from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'data/weights/best_openvino_model'), 
         glob('data/weights/best_openvino_model/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='balakarthik',
    maintainer_email='bkkarthik1599@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_perception_node = robot_perception.yolo_perception_node:main',
            'collect_training_images = robot_perception.collect_training_images:main',
        ],
    },
)
