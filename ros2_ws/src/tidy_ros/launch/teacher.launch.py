"""Run the simulator and the scripted teacher for a batch of seeds.

    ros2 launch tidy_ros teacher.launch.py seed_start:=0 num_episodes:=10
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    client = None
    ld = LaunchDescription(
        [
            DeclareLaunchArgument("seed_start", default_value="0"),
            DeclareLaunchArgument("num_episodes", default_value="5"),
            DeclareLaunchArgument("image_size", default_value="128"),
            DeclareLaunchArgument("publish_video", default_value="false"),
            DeclareLaunchArgument("record", default_value="false", description="also run recorder_node"),
            DeclareLaunchArgument("repo_id", default_value="takerunakao/tidy-ros"),
            DeclareLaunchArgument("root", default_value="/ws/data/tidy-ros"),
            Node(
                package="tidy_ros",
                executable="sim_node",
                name="sim_node",
                output="screen",
                parameters=[
                    {
                        "image_size": LaunchConfiguration("image_size"),
                        "publish_video": LaunchConfiguration("publish_video"),
                    }
                ],
            ),
            client := Node(
                package="tidy_ros",
                executable="teacher_node",
                name="teacher_node",
                output="screen",
                parameters=[
                    {
                        "seed_start": LaunchConfiguration("seed_start"),
                        "num_episodes": LaunchConfiguration("num_episodes"),
                    }
                ],
            ),
            Node(
                package="tidy_ros",
                executable="recorder_node",
                name="recorder_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("record")),
                parameters=[
                    {
                        "repo_id": LaunchConfiguration("repo_id"),
                        "root": LaunchConfiguration("root"),
                        "image_size": LaunchConfiguration("image_size"),
                        "overwrite": True,
                    }
                ],
            ),
        ]
    )
    # The simulator is a server; stop everything once the client has finished its seeds.
    ld.add_action(RegisterEventHandler(OnProcessExit(target_action=client, on_exit=[Shutdown()])))
    return ld
