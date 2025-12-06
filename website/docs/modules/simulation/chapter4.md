---
sidebar_position: 5
---

# Chapter 4: Integrating Gazebo and Unity

## Learning Objectives

- Understand how to combine Gazebo and Unity for comprehensive robot simulation
- Learn architectural approaches for multi-platform simulation
- Implement data synchronization between Gazebo and Unity
- Create seamless workflows that leverage the strengths of both platforms
- Design hybrid simulation environments for humanoid robots

## Introduction to Multi-Platform Simulation

Using both Gazebo and Unity in a single robotics pipeline allows us to leverage the strengths of each platform:

- **Gazebo**: Excellent physics simulation, established ROS integration, sensor modeling
- **Unity**: High-fidelity rendering, realistic visualization, advanced graphics capabilities

The integration of both platforms enables a complete simulation pipeline where physics-correct simulation occurs in Gazebo while high-fidelity visualization and human interaction happen in Unity.

## Approaches to Integration

### 1. Parallel Simulation with Synchronization

Run both simulators independently but keep their states synchronized:

```
[Robot Controller] → [Gazebo (Physics)] ↔ [Synchronization Layer] ↔ [Unity (Rendering)]
                      ↓                    ↓                           ↓
                [Physics Results]    [State Update]              [Visualization]
```

### 2. Specialized Task Allocation

- **Gazebo**: Physics simulation, sensor data generation, navigation planning
- **Unity**: High-fidelity rendering, human-robot interaction, realistic scene visualization

### 3. Hybrid Architecture

Use a middleware layer to manage data flow between both simulators:

```
[ROS 2 Nodes]
     |
[Middleware Layer] 
     |---------|---------|
     |         |         |
[Gazebo]  [Unity]  [Other Tools]
   (Physics) (Rendering)
```

## Implementation Architecture

### Synchronization Layer

The synchronization layer is critical for maintaining consistency between both simulators:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Twist
import time
import threading

class SimulationSynchronizer(Node):
    def __init__(self):
        super().__init__('simulation_synchronizer')
        
        # Publishers for both simulators
        self.gazebo_joint_pub = self.create_publisher(JointState, '/gazebo/joint_commands', 10)
        self.unity_joint_pub = self.create_publisher(JointState, '/unity/joint_commands', 10)
        
        # Subscribers from both simulators
        self.gazebo_state_sub = self.create_subscription(
            JointState, '/gazebo/joint_states', self.gazebo_state_callback, 10)
        self.unity_state_sub = self.create_subscription(
            JointState, '/unity/joint_states', self.unity_state_callback, 10)
        
        # Sync timer
        self.sync_timer = self.create_timer(0.01, self.sync_states)  # 100 Hz sync
        
        # State storage
        self.gazebo_state = JointState()
        self.unity_state = JointState()
        self.last_sync_time = time.time()
        
        self.get_logger().info('Simulation Synchronizer initialized')
    
    def gazebo_state_callback(self, msg):
        """Handle joint states from Gazebo"""
        self.gazebo_state = msg
        # Forward to Unity for visualization
        self.unity_joint_pub.publish(msg)
    
    def unity_state_callback(self, msg):
        """Handle joint states from Unity visualization"""
        self.unity_state = msg
        # Could forward to Gazebo if needed
        # In typical setup, Unity is "slave" to physics simulation
    
    def sync_states(self):
        """Synchronize states between simulators"""
        # In this example, Gazebo is authoritative for physics
        # Unity follows Gazebo's state for visualization
        
        # Check for significant desync (e.g., due to network delay)
        if self.gazebo_state.name and self.unity_state.name:
            if len(self.gazebo_state.position) == len(self.unity_state.position):
                max_diff = 0
                for i in range(len(self.gazebo_state.position)):
                    diff = abs(self.gazebo_state.position[i] - self.unity_state.position[i])
                    max_diff = max(max_diff, diff)
                
                if max_diff > 0.1:  # Threshold for significant desync
                    self.get_logger().warn(f'Significant desync detected: {max_diff} rad')
        
        # Keep Unity in sync with Gazebo state
        if self.gazebo_state.name and self.gazebo_state.position:
            sync_msg = JointState()
            sync_msg.header.stamp = self.get_clock().now().to_msg()
            sync_msg.name = self.gazebo_state.name
            sync_msg.position = self.gazebo_state.position
            sync_msg.velocity = self.gazebo_state.velocity
            sync_msg.effort = self.gazebo_state.effort
            
            # Send to Unity to update visualization
            self.unity_joint_pub.publish(sync_msg)

def main(args=None):
    rclpy.init(args=args)
    synchronizer = SimulationSynchronizer()
    
    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        pass
    finally:
        synchronizer.destroy_node()
        rclpy.shutdown()
```

## Advanced Synchronization Techniques

### State Prediction for Network Delays

When there are network delays between simulators, we can use prediction:

```python
import numpy as np
from collections import deque

class PredictiveSynchronizer:
    def __init__(self):
        self.state_history = deque(maxlen=10)  # Store last 10 states
        self.network_delay_estimate = 0.05  # 50ms delay estimate
        
    def add_state(self, state, timestamp):
        """Add state to history for prediction"""
        self.state_history.append({
            'state': state,
            'timestamp': timestamp
        })
    
    def predict_state(self, target_time):
        """Predict state at a future time"""
        if len(self.state_history) < 2:
            return None
        
        # Simple linear extrapolation between last two states
        recent_states = list(self.state_history)[-2:]
        
        if len(recent_states) < 2:
            return recent_states[0]['state']
        
        state1 = recent_states[0]
        state2 = recent_states[1]
        
        dt = state2['timestamp'] - state1['timestamp']
        if dt <= 0:
            return state2['state']
        
        # Calculate velocity (simplified for joint positions)
        dt_target = target_time - state2['timestamp']
        predicted_state = JointState()
        predicted_state.name = state2['state'].name
        predicted_state.position = []
        
        for i in range(len(state2['state'].position)):
            if i < len(state1['state'].position):
                velocity = (state2['state'].position[i] - state1['state'].position[i]) / dt
                predicted_pos = state2['state'].position[i] + velocity * dt_target
                predicted_state.position.append(predicted_pos)
            else:
                predicted_state.position.append(state2['state'].position[i])
        
        return predicted_state
```

## Data Exchange Patterns

### 1. Sensor Data Exchange

```python
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry

class SensorDataExchange:
    def __init__(self, node):
        self.node = node
        
        # Publishers to both simulators
        self.gazebo_sensor_pub = node.create_publisher(LaserScan, '/gazebo/scanner', 10)
        self.unity_sensor_pub = node.create_publisher(Image, '/unity/camera', 10)
        
        # Subscribers from both simulators
        self.gazebo_odom_sub = node.create_subscription(
            Odometry, '/gazebo/odom', self.gazebo_odom_callback, 10)
        self.unity_imu_sub = node.create_subscription(
            Imu, '/unity/imu', self.unity_imu_callback, 10)
        
        # Synchronization flag
        self.odom_sync_enabled = True
        self.imu_sync_enabled = True
    
    def gazebo_odom_callback(self, msg):
        """Forward Gazebo odometry to Unity for visualization"""
        if self.odom_sync_enabled:
            # Convert odometry to transform for Unity
            # This would typically send pose data to Unity visualization
            pass
    
    def unity_imu_callback(self, msg):
        """Forward Unity IMU to Gazebo for simulation"""
        if self.imu_sync_enabled:
            # In a real implementation, Unity might send IMU data back to Gazebo
            # for consistency between visual and physical simulation
            pass
```

### 2. Control Command Distribution

```python
class ControlDistribution:
    def __init__(self, node):
        self.node = node
        
        # Subscriber for high-level commands
        self.cmd_sub = node.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10)
        
        # Publishers to both simulators
        self.gazebo_cmd_pub = node.create_publisher(Twist, '/gazebo/cmd_vel', 10)
        self.unity_cmd_pub = node.create_publisher(Twist, '/unity/cmd_vel', 10)
        
        # Control parameters
        self.distribution_mode = "duplicate"  # duplicate or split by function
    
    def command_callback(self, msg):
        """Distribute commands to both simulators"""
        if self.distribution_mode == "duplicate":
            # Send the same command to both simulators
            self.gazebo_cmd_pub.publish(msg)
            self.unity_cmd_pub.publish(msg)
        elif self.distribution_mode == "function_split":
            # Different parts of the command go to different simulators
            # e.g., locomotion to Gazebo, head movement to Unity
            self.distribute_by_function(msg)
    
    def distribute_by_function(self, cmd):
        """Distribute command based on function"""
        # Send translational movement to Gazebo (physics)
        gazebo_cmd = Twist()
        gazebo_cmd.linear.x = cmd.linear.x
        gazebo_cmd.linear.y = cmd.linear.y
        gazebo_cmd.angular.z = cmd.angular.z  # Turning
        self.gazebo_cmd_pub.publish(gazebo_cmd)
        
        # Send head movement to Unity (visualization)
        unity_cmd = Twist()
        unity_cmd.angular.x = cmd.angular.x  # Neck pitch
        unity_cmd.angular.y = cmd.angular.y  # Neck yaw
        self.unity_cmd_pub.publish(unity_cmd)
```

## Real-World Implementation Example

### Launch File for Integrated Simulation

```xml
<!-- integrated_simulation.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo simulation
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
                'worlds',
                'humanoid_world.world'
            ])
        }.items()
    )
    
    # Launch Unity simulation (this would connect via TCP)
    unity_node = Node(
        package='unity_simulation',
        executable='unity_bridge',
        name='unity_bridge',
        parameters=[
            {'unity_ip': '127.0.0.1'},
            {'unity_port': 10000}
        ]
    )
    
    # Launch synchronization node
    sync_node = Node(
        package='simulation_integration',
        executable='simulation_synchronizer',
        name='simulation_synchronizer'
    )
    
    # Launch sensor data exchange
    sensor_exchange_node = Node(
        package='simulation_integration',
        executable='sensor_exchange',
        name='sensor_exchange'
    )
    
    # Launch control distribution
    control_dist_node = Node(
        package='simulation_integration',
        executable='control_distribution',
        name='control_distribution'
    )
    
    return LaunchDescription([
        gazebo_launch,
        unity_node,
        sync_node,
        sensor_exchange_node,
        control_dist_node
    ])
```

## Visualization Data Pipeline

### Converting Gazebo Data for Unity

```csharp
// Unity C# script to receive data from Gazebo via ROS-TCP-Connector
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class GazeboDataReceiver : MonoBehaviour
{
    ROSConnection ros;
    
    public Transform robotRoot;
    public Transform[] jointTransforms;
    public string[] jointNames;
    
    void Start()
    {
        ros = ROSConnection.instance;
        ros.Subscribe<JointStateMsg>("/gazebo/joint_states", JointStateCallback);
    }
    
    void JointStateCallback(JointStateMsg jointState)
    {
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];
            double position = jointState.position[i];
            
            // Find corresponding joint in Unity model
            for (int j = 0; j < jointNames.Length; j++)
            {
                if (jointNames[j] == jointName && j < jointTransforms.Length)
                {
                    // Apply position to Unity joint (convert from radians to degrees)
                    jointTransforms[j].localEulerAngles = new Vector3(
                        0, 
                        (float)(position * Mathf.Rad2Deg), 
                        0
                    );
                    break;
                }
            }
        }
        
        // Update robot position and orientation
        // This would come from a separate pose topic in a complete implementation
    }
}
```

## Performance Considerations

### Optimization Strategies

1. **Selective Synchronization**: Only sync states that are critical for visual consistency
2. **Data Compression**: Reduce data size through quantization or sampling
3. **Asynchronous Processing**: Use threading to prevent blocking operations
4. **Caching**: Cache frequently accessed transformations

```python
class OptimizedSynchronizer:
    def __init__(self):
        self.last_sync_values = {}  # Cache last synced values
        self.sync_threshold = 0.01  # Only sync if change exceeds threshold
        self.compression_factor = 4  # Reduce update frequency
    
    def should_sync_joint(self, joint_name, new_value):
        """Determine if a joint needs synchronization"""
        if joint_name not in self.last_sync_values:
            return True
        
        old_value = self.last_sync_values[joint_name]
        return abs(new_value - old_value) > self.sync_threshold
    
    def compress_state(self, joint_state):
        """Compress joint state data"""
        compressed = JointState()
        compressed.header = joint_state.header
        
        # Only include joints that have changed significantly
        for i, name in enumerate(joint_state.name):
            if i < len(joint_state.position):
                if self.should_sync_joint(name, joint_state.position[i]):
                    compressed.name.append(name)
                    compressed.position.append(joint_state.position[i])
                    self.last_sync_values[name] = joint_state.position[i]
        
        return compressed
```

## Debugging Multi-Platform Simulations

### Diagnostic Tools

```python
class SimulationDiagnostics(Node):
    def __init__(self):
        super().__init__('simulation_diagnostics')
        
        # Publishers for diagnostic data
        self.diag_pub = self.create_publisher(String, '/simulation_diagnostics', 10)
        
        # Diagnostic timer
        self.diag_timer = self.create_timer(5.0, self.publish_diagnostics)
        
        # State tracking
        self.gazebo_active = True
        self.unity_active = True
        self.sync_deviation = 0.0
        self.network_latency = 0.0
    
    def publish_diagnostics(self):
        """Publish diagnostic information"""
        diag_msg = String()
        diag_msg.data = f"""
        Simulation Diagnostics:
        - Gazebo Active: {self.gazebo_active}
        - Unity Active: {self.unity_active}
        - Sync Deviation: {self.sync_deviation:.3f} rad
        - Network Latency: {self.network_latency:.3f} ms
        - Last Sync: {time.time()}
        """
        
        self.diag_pub.publish(diag_msg)
```

## Use Cases for Integrated Simulation

### 1. Human-Robot Interaction Studies

- Physics-correct robot behavior in Gazebo
- Realistic human avatars and environments in Unity
- Accurate sensor simulation in Gazebo for perception
- High-fidelity rendering for human participants

### 2. Training AI Models

- Physics-accurate simulation for control training (Gazebo)
- High-fidelity rendering for perception training (Unity)
- Synthetic data generation with ground truth from both platforms

### 3. System Integration Testing

- Complete robot software stack with both simulators
- Validation of perception-action loops
- Human-in-the-loop testing with realistic interfaces

## Troubleshooting Common Issues

1. **Desynchronization**: Implement state prediction and correction
2. **Network Latency**: Optimize data transmission and add buffering
3. **Performance**: Use selective synchronization and data compression
4. **Data Format Mismatch**: Ensure consistent units and coordinate frames

## Summary

This chapter covered the integration of Gazebo and Unity for comprehensive humanoid robot simulation:

- Architectural approaches for multi-platform simulation
- Implementation of synchronization layers between simulators
- Advanced techniques for data exchange and performance optimization
- Real-world implementation examples and use cases
- Debugging and troubleshooting strategies for integrated simulations

The integration of Gazebo and Unity provides a powerful platform for humanoid robot development, combining accurate physics simulation with high-fidelity visualization. This approach enables more comprehensive testing and development of humanoid robots in a realistic virtual environment.

## Exercises

1. Set up a simple synchronization node between two simulated robots
2. Create a launch file that starts both Gazebo and Unity with synchronization
3. Implement a compression algorithm for joint state data

## Next Steps

With a complete understanding of physics simulation, sensor modeling, high-fidelity rendering, and simulation integration, you're now ready to explore the NVIDIA Isaac ecosystem in the next module. Isaac provides GPU-accelerated perception and AI capabilities that complement the simulation foundation you've built.