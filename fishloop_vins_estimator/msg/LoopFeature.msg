int32 feature_id
int32 camera_id
int32 view_id

geometry_msgs/Point32 point_w
geometry_msgs/Point32 bearing
geometry_msgs/Point32 raw_uv
geometry_msgs/Point32 virtual_uv

# Filled by loop_fusion after projecting the bearing into its canonical view.
# Keeping it in the transport contract makes saved/forwarded keyframes able to
# carry descriptors without changing the feature identity schema again.
uint8[] descriptor
