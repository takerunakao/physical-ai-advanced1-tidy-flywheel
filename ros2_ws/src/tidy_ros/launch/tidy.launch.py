"""Language-instructed tidy-up of the whole table as ROS 2 nodes (the ROS 2 counterpart of scripts/tidy_all.py).

    ros2 launch tidy_ros tidy.launch.py checkpoint:=/ws/outputs/act_m3-v0/checkpoints/last/pretrained_model \\
        seeds:=1000-1001 video_dir:=/ws/outputs/ros_tidy_videos results_json:=/ws/outputs/tidy_all/ros_m3-v0.json

sim_node (multi scene, all objects on the table) + vlm_node (Claude planner/pointer; needs ANTHROPIC_API_KEY)
+ policy_node (mode serve: /policy/execute) + tidy_node (orchestrator) [+ video_node with the status overlay].
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    client = None
    recording = PythonExpression(["'", LaunchConfiguration("video_dir"), "' != ''"])
    ld = LaunchDescription(
        [
            DeclareLaunchArgument("checkpoint"),
            DeclareLaunchArgument("seeds", default_value="1000-1009"),
            DeclareLaunchArgument("command", default_value="片付けて"),
            DeclareLaunchArgument("max_subtasks", default_value="6"),
            DeclareLaunchArgument("snap_drop", default_value="true"),
            DeclareLaunchArgument("scene", default_value="multi"),
            DeclareLaunchArgument("tag", default_value="tidy"),
            DeclareLaunchArgument("results_json", default_value=""),
            DeclareLaunchArgument("video_dir", default_value="", description="if set, run video_node into this dir"),
            Node(
                package="tidy_ros",
                executable="sim_node",
                name="sim_node",
                output="screen",
                parameters=[{"scene": LaunchConfiguration("scene"), "prefill_p": 0.0, "publish_video": recording}],
            ),
            Node(package="tidy_ros", executable="vlm_node", name="vlm_node", output="screen"),
            Node(
                package="tidy_ros",
                executable="policy_node",
                name="policy_node",
                output="screen",
                parameters=[{"mode": "serve", "checkpoint": LaunchConfiguration("checkpoint")}],
            ),
            client := Node(
                package="tidy_ros",
                executable="tidy_node",
                name="tidy_node",
                output="screen",
                parameters=[
                    {
                        "command": ParameterValue(LaunchConfiguration("command"), value_type=str),
                        "seeds": ParameterValue(LaunchConfiguration("seeds"), value_type=str),  # "1000" stays a string
                        "max_subtasks": LaunchConfiguration("max_subtasks"),
                        "snap_drop": LaunchConfiguration("snap_drop"),
                        "results_json": LaunchConfiguration("results_json"),
                        "executor": LaunchConfiguration("checkpoint"),
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
    # The simulator, VLM and policy are servers; stop everything once the orchestrator has finished its seeds.
    ld.add_action(RegisterEventHandler(OnProcessExit(target_action=client, on_exit=[Shutdown()])))
    return ld
