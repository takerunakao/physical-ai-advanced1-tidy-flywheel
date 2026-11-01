"""Run the simulator and a learned policy on held-out seeds.

    ros2 launch tidy_ros policy.launch.py checkpoint:=/ws/outputs/act_v0 seeds:=1000-1019
    # + one 640x480 MP4 per episode, recorded from the ROS 2 camera topic by video_node
    ros2 launch tidy_ros policy.launch.py checkpoint:=... seeds:=1000-1004 video_dir:=/ws/outputs/ros_videos
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    client = None
    recording = PythonExpression(["'", LaunchConfiguration("video_dir"), "' != ''"])
    ld = LaunchDescription(
        [
            DeclareLaunchArgument("checkpoint"),
            DeclareLaunchArgument("seeds", default_value="1000-1009"),
            DeclareLaunchArgument("tag", default_value="policy"),
            DeclareLaunchArgument("results_json", default_value=""),
            DeclareLaunchArgument("video_dir", default_value="", description="if set, run video_node into this dir"),
            Node(
                package="tidy_ros",
                executable="sim_node",
                name="sim_node",
                output="screen",
                parameters=[{"publish_video": recording}],
            ),
            client := Node(
                package="tidy_ros",
                executable="policy_node",
                name="policy_node",
                output="screen",
                parameters=[
                    {
                        "checkpoint": LaunchConfiguration("checkpoint"),
                        "seeds": LaunchConfiguration("seeds"),
                        "tag": LaunchConfiguration("tag"),
                        "results_json": LaunchConfiguration("results_json"),
                    }
                ],
            ),
            Node(
                package="tidy_ros",
                executable="video_node",
                name="video_node",
                output="screen",
                parameters=[{"video_dir": LaunchConfiguration("video_dir"), "tag": LaunchConfiguration("tag")}],
                condition=IfCondition(recording),
            ),
        ]
    )
    # The simulator is a server; stop everything once the client has finished its seeds.
    ld.add_action(RegisterEventHandler(OnProcessExit(target_action=client, on_exit=[Shutdown()])))
    return ld
