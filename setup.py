from setuptools import find_packages, setup

package_name = 'metro_lidar' # Название вашего текущего ROS 2 пакета

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andreymazo',
    maintainer_email='andreymazo@todo.todo',
    description='Autonomous Subway 3D Lidar Vision Core Node',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 🟢 ГЛАВНЫЙ ИИ-МОСТ: связываем консольную команду запуска с кодом ноды
             'subway_vision_node = metro_lidar.lidar_detector_node:main'
        ],
    },
)
