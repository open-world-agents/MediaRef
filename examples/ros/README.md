# ROS MediaRef Tools

Tools for working with ROS bags and MediaRef format.

## Scripts

- **`bag_info.py`** - Display bag contents (ROS1/ROS2/MCAP)
- **`bag_to_mediaref.py`** - Convert bags with CompressedImage to MediaRef format
- **`mediaref_decode.py`** - Read and decode MediaRef bags

## Installation

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or install globally
pip3 install -r requirements.txt
```

## Usage

### View bag contents
```bash
# Auto-detects format (ROS1/ROS2)
uv run bag_info.py ba234b52c88d7f1f0da04baab375f574.bag
uv run bag_info.py ba234b52c88d7f1f0da04baab375f574/

# Show more messages per topic
uv run bag_info.py my_bag.bag -n 5
```

### Convert to MediaRef format
```bash
# Auto-detects ROS1/ROS2
uv run bag_to_mediaref.py input.bag
uv run bag_to_mediaref.py input/

# Custom output path and video settings
uv run bag_to_mediaref.py input.bag -o output.bag --fps 30 --keyframe-interval 1.0
```

### Read MediaRef bags
```bash
# Display MediaRef messages
uv run mediaref_decode.py output_mediaref.bag -n 10

# Batch decode and save frames
uv run mediaref_decode.py output_mediaref.bag --batch-decode -n 20 -o frames/
```

## Bag Info

```sh
rosbag info ba234b52c88d7f1f0da04baab375f574.bag

path:        ba234b52c88d7f1f0da04baab375f574.bag
version:     2.0
duration:    45.0s
start:       Oct 22 2025 01:32:44.52 (1761096764.52)
end:         Oct 22 2025 01:33:29.54 (1761096809.54)
size:        63.3 MB
messages:    13132
compression: none [82/82 chunks]
types:       geometry_msgs/PointStamped  [c63aecb41bfdfd6b7e1fac37c7cbe7bf]
             geometry_msgs/TwistStamped  [98d34b0043a2093cf9d9345ab6eef12e]
             nav_msgs/Odometry           [cd5e73d190d741a2f92e81eda573aca7]
             sensor_msgs/CompressedImage [8f7a12909da2c9d3332d540a0977563f]
             sensor_msgs/Imu             [6a62c6daae103f4ff57a132d6f95cec2]
             sensor_msgs/Joy             [5a9ea5f83505693b71e785041e67a8bb]
             sensor_msgs/TimeReference   [fded64a0265108ba86c3d38fb11c0c16]
             std_msgs/Bool               [8b94c1b53db61fb6aed406028ad6332a]
topics:      /cmd_vel                             901 msgs    : geometry_msgs/TwistStamped
             /cmd_vel/latency                     901 msgs    : sensor_msgs/TimeReference
             /imu                                 901 msgs    : sensor_msgs/Imu
             /is_model                            900 msgs    : std_msgs/Bool
             /joy                                1940 msgs    : sensor_msgs/Joy
             /local_map/local_state/compressed    450 msgs    : sensor_msgs/CompressedImage
             /local_map/point_stamped             450 msgs    : geometry_msgs/PointStamped
             /odom                                900 msgs    : nav_msgs/Odometry
             /odom/latency                        900 msgs    : sensor_msgs/TimeReference
             /rgb_front/compressed                653 msgs    : sensor_msgs/CompressedImage
             /rgb_front/latency                   653 msgs    : sensor_msgs/TimeReference
             /rgb_left/compressed                 672 msgs    : sensor_msgs/CompressedImage
             /rgb_left/latency                    672 msgs    : sensor_msgs/TimeReference
             /rgb_right/compressed                670 msgs    : sensor_msgs/CompressedImage
             /rgb_right/latency                   670 msgs    : sensor_msgs/TimeReference
             /wheel_odom                          899 msgs    : nav_msgs/Odometry
```