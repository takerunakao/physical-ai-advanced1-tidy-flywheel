from setuptools import find_packages, setup
from glob import glob

package_name = "tidy_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="takerunakao",
    maintainer_email="57853518+takerunakao@users.noreply.github.com",
    description="ROS 2 nodes for the tidy-flywheel project",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_node = tidy_ros.sim_node:main",
            "teacher_node = tidy_ros.teacher_node:main",
            "policy_node = tidy_ros.policy_node:main",
            "recorder_node = tidy_ros.recorder_node:main",
            "video_node = tidy_ros.video_node:main",
            "vlm_node = tidy_ros.vlm_node:main",
            "tidy_node = tidy_ros.tidy_node:main",
        ],
    },
)
