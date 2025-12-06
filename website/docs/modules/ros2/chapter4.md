---
sidebar_position: 5
---

# Chapter 4: URDF and Humanoid Robot Models

## Learning Objectives

- Understand the Unified Robot Description Format (URDF) for robot modeling
- Create URDF files for complex humanoid robot structures
- Learn about joints, links, inertial properties, and visual/collision elements
- Integrate URDF models with ROS 2 simulation environments
- Validate and test humanoid robot models

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. It defines the physical structure of a robot including links, joints, inertial properties, visual representations, and collision models. For humanoid robots, URDF is essential for simulation, visualization, and motion planning.

### Why URDF is Important for Humanoid Robots

1. **Simulation**: Gazebo and other simulators use URDF to create physics models
2. **Visualization**: RViz uses URDF to display robots in 3D
3. **Motion Planning**: Planning algorithms need URDF models to understand robot kinematics
4. **Control**: Robot controllers use URDF for kinematic calculations
5. **Standardization**: URDF provides a standard way to model robots across the ROS community

## URDF Structure Overview

A URDF file consists of:
- **Links**: Rigid parts of the robot (e.g., torso, limb segments)
- **Joints**: Connections between links (e.g., hinges, prismatic joints)
- **Materials**: Visual properties (colors, textures)
- **Gazebos**: Simulation-specific properties

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
  
  <!-- Connected link via joint -->
  <link name="upper_body">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Joint connecting the two links -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="upper_body"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>
</robot>
```

## Links in URDF

Links represent rigid bodies in the robot. Each link contains:

1. **Visual Properties**: How the link appears in visualization
2. **Collision Properties**: How the link interacts in physics simulation
3. **Inertial Properties**: Physical properties for physics simulation

### Visual and Collision Elements

```xml
<link name="arm_link">
  <!-- Visual properties for display -->
  <visual>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://robot_meshes/arm.dae"/>
    </geometry>
    <material name="gray">
      <color rgba="0.5 0.5 0.5 1"/>
    </material>
  </visual>
  
  <!-- Collision properties for physics -->
  <collision>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.2" radius="0.05"/>
    </geometry>
  </collision>
  
  <!-- Inertial properties -->
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
  </inertial>
</link>
```

## Joints in URDF

Joints define how links connect and move relative to each other. Common joint types:

1. **Fixed**: No movement (welded joint)
2. **Revolute**: Rotational movement around single axis (like hinges)
3. **Continuous**: Like revolute but unlimited rotation
4. **Prismatic**: Linear sliding movement
5. **Floating**: 6DOF movement
6. **Planar**: Movement in a plane

### Joint Definitions

```xml
<!-- Revolute joint example: hip pitch -->
<joint name="left_hip_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0 -0.1 -0.1" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>  <!-- Rotation around X-axis -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Continuous joint example: neck rotation -->
<joint name="neck_yaw" type="continuous">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.8" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Rotation around Z-axis -->
  <dynamics damping="0.1"/>
</joint>

<!-- Fixed joint example: sensor mount -->
<joint name="imu_mount" type="fixed">
  <parent link="head"/>
  <child link="imu_link"/>
  <origin xyz="0.05 0 0.02" rpy="0 0 0"/>
</joint>
```

## Creating a Humanoid Robot Model

Let's create a simplified humanoid model with legs, torso, arms, and head:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://ros.org/wiki/xacro">
  <!-- Include common properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  
  <!-- Material definitions -->
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.2 0.2 1"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 0.8 1"/>
  </material>

  <!-- Base link: pelvis/hip -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.15 0.2 0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.2 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Left leg chain -->
  <joint name="left_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0 -0.1 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100" velocity="3.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <capsule length="0.35" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.35" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="3.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <capsule length="0.35" radius="0.04"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.35" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.0015"/>
    </inertial>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right leg (similar structure) -->
  <joint name="right_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0 0.1 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="100" velocity="3.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <geometry>
        <capsule length="0.35" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.35" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="3.0"/>
  </joint>

  <link name="right_shin">
    <visual>
      <geometry>
        <capsule length="0.35" radius="0.04"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.35" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.0015"/>
    </inertial>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.4"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.785" upper="0.785" effort="10" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Left arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.05 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <capsule length="0.25" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.35" upper="0" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <capsule length="0.25" radius="0.035"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.25" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.7"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_wrist" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2.0"/>
  </joint>

  <link name="left_hand">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <!-- Right arm (similar structure) -->
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.05 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <capsule length="0.25" radius="0.04"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.35" upper="0" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <capsule length="0.25" radius="0.035"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.25" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.7"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_wrist" type="revolute">
    <parent link="right_lower_arm"/>
    <child link="right_hand"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2.0"/>
  </joint>

  <link name="right_hand">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>
</robot>
```

## Xacro for Complex Models

Xacro (XML Macros) is a macro language that extends URDF with features like variables, constants, and macros to simplify complex robot models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="humanoid_with_xacro">
  
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.4" />
  <xacro:property name="torso_width" value="0.2" />
  <xacro:property name="torso_depth" value="0.2" />
  
  <!-- Macro for defining a leg -->
  <xacro:macro name="leg" params="side reflect">
    <joint name="${side}_hip_yaw" type="revolute">
      <parent link="base_link"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 ${reflect * 0.1} -0.05" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-0.785" upper="0.785" effort="100" velocity="3.0"/>
    </joint>

    <link name="${side}_thigh">
      <visual>
        <geometry>
          <capsule length="0.35" radius="0.05"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <capsule length="0.35" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${side}_knee" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="2.35" effort="100" velocity="3.0"/>
    </joint>

    <link name="${side}_shin">
      <visual>
        <geometry>
          <capsule length="0.35" radius="0.04"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <capsule length="0.35" radius="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.0015"/>
      </inertial>
    </link>

    <joint name="${side}_ankle" type="revolute">
      <parent link="${side}_shin"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="0.5" effort="50" velocity="2.0"/>
    </joint>

    <link name="${side}_foot">
      <visual>
        <geometry>
          <box size="0.15 0.08 0.05"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.15 0.08 0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.15 0.2 0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.2 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>
  
  <!-- Use the leg macro to define both legs -->
  <xacro:leg side="left" reflect="-1" />
  <xacro:leg side="right" reflect="1" />
  
  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## URDF for Simulation with Gazebo

To use URDF in Gazebo simulation, we need to add Gazebo-specific tags:

```xml
<!-- Gazebo material definition -->
<gazebo reference="base_link">
  <material>Gazebo/White</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>

<!-- Gazebo plugin for ros_control -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>

<!-- Joint transmission for ros_control -->
<transmission name="left_hip_yaw_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_hip_yaw">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_yaw_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Validating URDF Models

### Tools for URDF Validation

1. **check_urdf**: Command-line tool to validate URDF syntax
   ```bash
   check_urdf /path/to/robot.urdf
   ```

2. **urdf_to_graphiz**: Generate a visual graph of the kinematic tree
   ```bash
   urdf_to_graphiz /path/to/robot.urdf
   ```

3. **RViz**: Visualize the robot model

### Common URDF Issues

1. **Incorrect Mass Values**: Too light or too heavy links
2. **Invalid Inertial Values**: Non-positive definite inertia matrices
3. **Undefined Materials**: Referenced materials that don't exist
4. **Joint Limit Issues**: Invalid or unrealistic joint limits
5. **Kinematic Loops**: URDF only supports tree structures, not loops

## Loading URDF in ROS 2

To use your URDF model in ROS 2, you need to:

1. **Launch the robot state publisher**:

```xml
<!-- robot_state_publisher_launch.py -->
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindPackageShare("your_robot_description"), "urdf", "robot.urdf.xacro"]),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    return LaunchDescription([
        node_robot_state_publisher,
    ])
```

2. **Using it in a ROS 2 node**:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )
        
        # TF broadcaster for robot transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz
        
        self.current_joint_states = JointState()
        self.get_logger().info('Humanoid Controller initialized')
    
    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.current_joint_states = msg
    
    def control_loop(self):
        """Main control loop"""
        # Example: move left arm in a simple pattern
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = ['left_shoulder_pitch', 'right_shoulder_pitch']
        
        # Generate a simple oscillating pattern
        t = self.get_clock().now().nanoseconds / 1e9  # Time in seconds
        left_pos = 0.5 * math.sin(t)  # Oscillate between -0.5 and 0.5
        right_pos = 0.5 * math.sin(t + math.pi)  # Out of phase
        
        cmd_msg.position = [left_pos, right_pos]
        
        self.joint_cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced URDF Techniques

### Using Mesh Files

For more detailed robot models, use mesh files instead of primitive shapes:

```xml
<link name="head_link">
  <visual>
    <geometry>
      <!-- Use a mesh file for detailed geometry -->
      <mesh filename="package://humanoid_description/meshes/head.dae" scale="1 1 1"/>
    </geometry>
    <material name="skin_color">
      <color rgba="0.96 0.87 0.70 1.0"/>
    </material>
  </visual>
  <collision>
    <!-- Use a simpler collision mesh for better performance -->
    <geometry>
      <mesh filename="package://humanoid_description/meshes/head_collision.stl" scale="1 1 1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
  </inertial>
</link>
```

### Gazebo-Specific Enhancements

```xml
<!-- Sensor definition in URDF for Gazebo -->
<gazebo reference="head_camera_frame">
  <sensor type="camera" name="head_camera">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>head_camera_frame</frame_name>
      <topic_name>image_raw</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

## Troubleshooting Common Issues

### 1. Robot Falls Through Ground
- Check inertial values (mass and inertia matrix)
- Verify joint limits and dynamics parameters
- Ensure proper collision geometries

### 2. Robot Explodes in Simulation
- Check mass values (not too low or negative)
- Verify inertia matrix is positive definite
- Check joint dynamics (damping and friction)

### 3. RViz Shows Incorrect Model
- Check for URDF syntax errors
- Verify that the robot_state_publisher is running
- Ensure joint states are being published correctly

## Summary

URDF is a fundamental component for representing humanoid robots in ROS 2. This chapter covered:

- The structure of URDF files with links, joints, visual, collision, and inertial properties
- How to model a complete humanoid robot with legs, torso, arms, and head
- The use of Xacro to simplify complex robot descriptions
- Integration with Gazebo simulation and ROS 2 systems
- Validation techniques and common troubleshooting approaches

Properly modeling your humanoid robot is crucial for successful simulation and control. The URDF serves as the bridge between the physical robot design and its digital representation in the ROS 2 ecosystem.

## Exercises

1. Create a URDF file for a simple humanoid robot with at least 10 joints
2. Visualize your robot model in RViz
3. Add a sensor (like a camera or IMU) to your robot model

## Next Steps

With a complete understanding of ROS 2 fundamentals, communication patterns, AI integration, and robot modeling, you're now ready to explore more advanced topics in humanoid robotics. The next module will cover physics simulation and sensor systems using Gazebo and other simulation tools.