# One lock-step transition: observation at time t and the action applied to it.
std_msgs/Header header
int64 seed
int32 t
sensor_msgs/Image front
sensor_msgs/Image wrist
float32[] state          # [tcp_x, tcp_y, tcp_z, tcp_yaw, gripper_opening]
float32[] action         # [dx, dy, dz, dyaw, grip]
float32[] cube_pose      # [x, y, z, yaw] (privileged)
bool grasped
