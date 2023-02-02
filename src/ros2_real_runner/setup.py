from setuptools import setup

package_name = 'ros2_real_runner'
sub_package_name1 = 'ros2_real_runner.PyTorch_YOLOv3.pytorchyolo'
sub_package_name2 = 'ros2_real_runner.PyTorch_YOLOv3.pytorchyolo.utils'
sub_package_name3 = 'ros2_real_runner.ros2_numpy.ros2_numpy'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, sub_package_name1, sub_package_name2, sub_package_name3],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seniorcar',
    maintainer_email='bq17088@shibaura-it.ac.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'real_runner = ros2_real_runner.real_main:main'
            
        ],
    },
)
