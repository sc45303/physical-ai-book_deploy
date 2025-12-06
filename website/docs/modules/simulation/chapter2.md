---
sidebar_position: 3
---

# Chapter 2: Sensors and Environment Building

## Learning Objectives

- Understand how to simulate various robot sensors in Gazebo
- Build realistic environments for robot testing
- Configure sensor models with appropriate parameters
- Integrate simulated sensors with ROS 2 systems
- Validate sensor data for humanoid robot applications

## Sensor Simulation in Robotics

Sensor simulation is a critical component of robot development, allowing for safe and cost-effective testing of perception algorithms. In humanoid robotics, accurate sensor simulation is particularly important due to the complex interaction between the robot and its environment.

### Types of Sensors in Humanoid Robots

Humanoid robots typically use these sensor types:

1. **Proprioceptive Sensors**: Measure internal robot state
   - Joint encoders: Position, velocity of joints
   - IMUs: Orientation, angular velocity, acceleration
   - Force/Torque sensors: Forces at joints and end effectors

2. **Exteroceptive Sensors**: Measure environment
   - Cameras: Visual perception
   - LIDAR: Distance measurements for navigation
   - Sonar: Additional distance sensing
   - Tactile sensors: Contact detection

## Gazebo Sensor Plugins

Gazebo provides plugins for simulating various sensors. These plugins publish data to ROS topics that can be processed by robot algorithms.

### Camera Sensors

Camera sensors in Gazebo simulate RGB cameras and publish images to ROS topics:

```xml
<!-- Example camera sensor in a URDF/SDF -->
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.089</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
    <topic_name>image_raw</topic_name>
  </plugin>
</sensor>
```

### LIDAR Sensors

LIDAR (Light Detection and Ranging) sensors simulate laser range finders:

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
    <topic_name>scan</topic_name>
    <frame_name>lidar_frame</frame_name>
  </plugin>
</sensor>
```

### IMU Sensors

IMU (Inertial Measurement Unit) sensors provide orientation and acceleration data:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <topic_name>imu</topic_name>
    <body_name>imu_link</body_name>
    <frame_name>imu_link</frame_name>
  </plugin>
</sensor>
```

## Environment Building in Gazebo

Creating realistic environments is crucial for meaningful robot testing. Gazebo provides several methods to build environments:

### World Files

World files define the complete simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="small_room">
    <!-- Include the sun -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Include the ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Add furniture -->
    <model name="table">
      <pose>-1 0 0 0 0 0</pose>
      <include>
        <uri>model://table</uri>
      </include>
    </model>
    
    <model name="chair">
      <pose>-1.5 0.5 0 0 0 1.57</pose>
      <include>
        <uri>model://chair</uri>
      </include>
    </model>
    
    <!-- Add objects for manipulation -->
    <model name="box">
      <pose>-0.8 0.3 0.5 0 0 0</pose>
      <link name="box_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <iyy>0.0001</iyy>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Building Complex Environments

For humanoid robots, environments should include:

- **Navigation areas**: Open spaces for walking, pathways
- **Obstacles**: Furniture, walls, other objects to navigate around
- **Interaction objects**: Items for manipulation tasks
- **Markers/landmarks**: Objects for localization and mapping
- **Varied terrain**: Different floor materials, slight inclines, stairs (for advanced robots)

## Sensor Integration with ROS 2

Once sensors are configured in Gazebo, they need to be integrated with ROS 2 systems:

### Camera Data Processing Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        
    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Process the image (example: detect edges)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Display the result
        cv2.imshow("Camera View", cv_image)
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    camera_processor = CameraProcessor()
    rclpy.spin(camera_processor)
    cv2.destroyAllWindows()
    camera_processor.destroy_node()
    rclpy.shutdown()
```

### LIDAR Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription  # prevent unused variable warning
        
    def scan_callback(self, msg):
        # Process LIDAR data
        # Convert to numpy array for easier processing
        ranges = np.array(msg.ranges)
        
        # Find minimum distance (closest obstacle)
        valid_ranges = ranges[np.isfinite(ranges)]  # Remove invalid (inf) values
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Closest obstacle: {min_distance:.2f}m')
        
        # Simple obstacle detection
        threshold = 1.0  # meters
        obstacles = valid_ranges < threshold
        obstacle_count = np.sum(obstacles)
        if obstacle_count > 0:
            self.get_logger().info(f'Found {obstacle_count} obstacles within {threshold}m')

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()
    rclpy.spin(lidar_processor)
    lidar_processor.destroy_node()
    rclpy.shutdown()
```

## Humanoid Robot Specific Sensors

Humanoid robots have unique sensor requirements:

### Balance Sensors

- **ZMP (Zero Moment Point) sensors**: Critical for bipedal stability
- **Force plates**: Measure ground reaction forces
- **Foot contact sensors**: Detect when feet make contact with ground

### Manipulation Sensors

- **Tactile sensors**: On fingertips for object manipulation
- **Force/Torque sensors**: In wrists to measure interaction forces
- **Stereo cameras**: For depth perception during manipulation

## Sensor Validation

Validating simulated sensors is crucial:

1. **Compare to real sensors**: When possible, compare simulated sensor data to real hardware
2. **Physics consistency**: Ensure sensor readings make sense given the simulated physics
3. **Timing accuracy**: Verify sensors publish at the correct rate
4. **Noise characteristics**: Ensure realistic noise models

## Example: Complete Sensor Setup for Humanoid Robot

```xml
<!-- Example link with multiple sensors -->
<link name="head">
  <visual>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.0083" ixy="0" ixz="0" iyy="0.0083" iyz="0" izz="0.0083"/>
  </inertial>
  
  <!-- Head camera -->
  <sensor name="head_camera" type="camera">
    <pose>0.05 0 0 0 0 0</pose>
    <camera name="head">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_frame</frame_name>
      <topic_name>head_camera/image_raw</topic_name>
    </plugin>
  </sensor>
  
  <!-- IMU in head -->
  <sensor name="head_imu" type="imu">
    <pose>0 0 0 0 0 0</pose>
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topic_name>imu/head</topic_name>
      <frame_name>head_imu_frame</frame_name>
    </plugin>
  </sensor>
</link>
```

## Troubleshooting Common Issues

### Sensor Not Publishing
- Check if the plugin is loaded correctly
- Verify topic names and namespaces
- Ensure the sensor has power/connections in the model

### Incorrect Sensor Data
- Verify sensor placement in the model
- Check coordinate frame transformations
- Validate sensor parameters (FOV, range, etc.)

### Performance Issues
- Reduce sensor update rates if not needed
- Lower image resolution for cameras
- Use fewer LIDAR rays if precision allows

## Summary

Sensor simulation is vital for developing and testing humanoid robots safely and efficiently. Proper configuration of sensor models, integration with ROS 2 systems, and validation of sensor data are crucial steps in creating realistic simulations. The environments you create should match the complexity of the real-world scenarios your humanoid robot will encounter.

## Exercises

1. Add a camera sensor to your simulated robot and visualize the output
2. Create a simple environment with obstacles for navigation testing
3. Implement a basic LIDAR obstacle detection node

## Next Steps

In the next chapter, we'll explore high-fidelity rendering and human-robot interaction using Unity, providing a different perspective on robot simulation and visualization.