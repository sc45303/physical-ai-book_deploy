---
sidebar_position: 4
---

# Chapter 3: Bridging Python Agents to ROS Controllers (rclpy)

## Learning Objectives

- Understand how to create Python nodes that interface with ROS 2
- Learn to use rclpy to communicate with ROS 2 systems
- Implement bridges between Python AI agents and ROS controllers
- Create robust communication patterns between AI and control systems
- Handle errors and exceptions in AI-control bridges

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing Python bindings for the ROS 2 middleware. It allows Python developers to create ROS 2 nodes and interact with the ROS 2 ecosystem. This is particularly important for humanoid robotics, where Python is widely used for AI and machine learning applications.

### Why Python for AI in Robotics

Python is the dominant language for AI and machine learning development due to:

- **Rich ecosystem**: Libraries like TensorFlow, PyTorch, scikit-learn, and OpenAI
- **Rapid prototyping**: Easy to develop and test AI algorithms
- **Community support**: Large community of AI researchers and practitioners
- **Integration capabilities**: Easy to integrate different systems and libraries

## Understanding the AI-Control Bridge

In humanoid robotics, there's often a need to bridge AI systems (running in Python) with robotic control systems (often using ROS 2). The bridge typically involves:

1. **Receiving sensor data** from ROS 2 topics
2. **Processing data** through AI algorithms
3. **Generating commands** based on AI decisions
4. **Sending commands** to robot controllers via ROS 2

### Architecture of AI-Control Bridge

```
[ROS 2 Sensors] → [Python Bridge Node] → [AI Agent] → [Python Bridge Node] → [ROS 2 Controllers]
```

## Setting up rclpy Nodes

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np

class AIBridgeNode(Node):
    def __init__(self):
        super().__init__('ai_bridge_node')
        
        # Publishers for sending commands to robot
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_publisher = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Subscribers for receiving sensor data
        self.sensor_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_subscriber = self.create_subscription(
            String,  # In practice, this would be sensor_msgs/Imu
            '/imu_data',
            self.imu_callback,
            10
        )
        
        # Store state data that will be processed by AI
        self.current_joint_states = JointState()
        self.imu_data = None
        
        # Timer for AI processing loop
        self.processing_timer = self.create_timer(0.1, self.process_ai_step)  # 10 Hz
        
        self.get_logger().info('AI Bridge Node initialized')
    
    def joint_state_callback(self, msg):
        """Process joint state messages"""
        self.current_joint_states = msg
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')
    
    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg.data
        self.get_logger().debug(f'Received IMU data: {msg.data}')
    
    def process_ai_step(self):
        """Process one step of AI algorithm"""
        # In a real system, this would call your AI agent
        # For now, we'll implement a simple balance controller
        
        if self.imu_data is not None:
            # Simple example: if robot is tilting, send corrective command
            try:
                # Convert string IMU data to numerical values
                tilt_angle = float(self.imu_data)
                
                if abs(tilt_angle) > 0.5:  # If tilting more than 0.5 radians
                    # Send corrective joint commands
                    cmd_msg = JointState()
                    cmd_msg.header.stamp = self.get_clock().now().to_msg()
                    cmd_msg.name = ['left_ankle_pitch', 'right_ankle_pitch']
                    cmd_msg.position = [-tilt_angle * 0.5, -tilt_angle * 0.5]  # Correcting torque
                    
                    self.joint_cmd_publisher.publish(cmd_msg)
                    self.get_logger().info(f'Sent corrective commands for tilt: {tilt_angle}')
            except ValueError:
                self.get_logger().error(f'Could not parse IMU data: {self.imu_data}')

def main(args=None):
    rclpy.init(args=args)
    ai_bridge_node = AIBridgeNode()
    
    try:
        rclpy.spin(ai_bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        ai_bridge_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementing AI Agent Integration

### Simple AI Agent Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

class SimpleAIAgent:
    def __init__(self):
        # In a real system, this might be a neural network or other ML model
        self.model = LinearRegression()
        self.is_trained = False
        
        # For humanoid balance control
        self.balance_history = []
        self.max_history = 100  # Store last 100 samples
    
    def predict_control(self, sensor_data):
        """
        Given sensor data, predict appropriate control actions
        sensor_data: dict containing sensor readings
        """
        if not self.is_trained:
            # For untrained model, return simple proportional control
            tilt = sensor_data.get('tilt', 0.0)
            return {'left_ankle_torque': -tilt * 0.5, 'right_ankle_torque': -tilt * 0.5}
        
        # Use trained model to predict control
        # This is a simplified example
        features = np.array([sensor_data['tilt'], sensor_data['angular_velocity']]).reshape(1, -1)
        control_output = self.model.predict(features)
        
        return {
            'left_ankle_torque': float(control_output[0]),
            'right_ankle_torque': float(control_output[1])
        }
    
    def add_training_data(self, sensor_data, control_output):
        """Add training data for future learning"""
        self.balance_history.append({
            'sensor': sensor_data.copy(),
            'control': control_output.copy()
        })
        
        # Keep only recent history
        if len(self.balance_history) > self.max_history:
            self.balance_history.pop(0)
    
    def train_model(self):
        """Train the model with collected data"""
        if len(self.balance_history) < 10:  # Need minimum data
            return False
        
        # Prepare training data
        X = []  # Sensor inputs
        y = []  # Control outputs
        
        for sample in self.balance_history:
            sensor_data = sample['sensor']
            control_data = sample['control']
            
            X.append([sensor_data['tilt'], sensor_data['angular_velocity']])
            y.append([control_data['left_ankle_torque'], control_data['right_ankle_torque']])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        return True
```

## Advanced Bridge Node with AI Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from builtin_interfaces.msg import Time
import numpy as np
import time

class AdvancedAIBridgeNode(Node):
    def __init__(self):
        super().__init__('advanced_ai_bridge_node')
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_publisher = self.create_publisher(JointState, '/joint_commands', 10)
        self.ai_feedback_publisher = self.create_publisher(Float32, '/ai_control_effort', 10)
        
        # Subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        self.force_torque_subscriber = self.create_subscription(
            String,  # In practice, this would use WrenchStamped or similar
            '/ft_sensors',
            self.ft_callback,
            10
        )
        
        # Initialize AI agent
        self.ai_agent = SimpleAIAgent()
        
        # State variables
        self.current_joint_states = JointState()
        self.current_imu = None
        self.ft_data = None
        self.last_control_time = time.time()
        
        # Processing timer
        self.processing_timer = self.create_timer(0.05, self.process_ai_step)  # 20 Hz
        
        # Training timer (slow, for continuous learning)
        self.training_timer = self.create_timer(5.0, self.train_model_if_needed)
        
        self.get_logger().info('Advanced AI Bridge Node initialized')
    
    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.current_joint_states = msg
    
    def imu_callback(self, msg):
        """Handle IMU data updates"""
        self.current_imu = msg
    
    def ft_callback(self, msg):
        """Handle force/torque sensor updates"""
        self.ft_data = msg.data  # Simplified string representation
    
    def process_ai_step(self):
        """Main AI processing step"""
        # Gather current sensor data
        sensor_data = self.get_sensor_data()
        
        if sensor_data is None:
            # Insufficient data to proceed
            return
        
        try:
            # Get AI prediction
            control_output = self.ai_agent.predict_control(sensor_data)
            
            # Execute control commands
            self.execute_control_commands(control_output)
            
            # Calculate and publish control effort feedback
            effort = self.calculate_control_effort(control_output)
            effort_msg = Float32()
            effort_msg.data = effort
            self.ai_feedback_publisher.publish(effort_msg)
            
            # Optionally store training data
            self.store_training_data(sensor_data, control_output)
            
            # Update timing for next control step
            self.last_control_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f'Error in AI processing: {str(e)}')
    
    def get_sensor_data(self):
        """Extract relevant sensor data for the AI agent"""
        if self.current_imu is None:
            return None
        
        # Extract relevant information from sensors
        sensor_data = {
            'tilt': self.current_imu.orientation.z,  # Simplified - in real systems, would use proper orientation
            'angular_velocity': self.current_imu.angular_velocity.z,
            'linear_acceleration': self.current_imu.linear_acceleration.x,
            'joint_positions': dict(zip(self.current_joint_states.name, self.current_joint_states.position)),
            'joint_velocities': dict(zip(self.current_joint_states.name, self.current_joint_states.velocity))
        }
        
        # Add time-based features
        dt = time.time() - self.last_control_time
        sensor_data['dt'] = dt
        
        return sensor_data
    
    def execute_control_commands(self, control_output):
        """Execute the control commands from the AI agent"""
        # Create joint command message
        joint_cmd_msg = JointState()
        joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Add commanded joint positions/torques
        for joint_name, torque_value in control_output.items():
            if 'torque' in joint_name:
                # This is a torque command
                joint_name_clean = joint_name.replace('_torque', '')
                joint_cmd_msg.name.append(joint_name_clean)
                joint_cmd_msg.effort.append(torque_value)
            elif 'position' in joint_name:
                # This is a position command
                joint_name_clean = joint_name.replace('_position', '')
                joint_cmd_msg.name.append(joint_name_clean)
                joint_cmd_msg.position.append(torque_value)
        
        # Publish joint commands
        if len(joint_cmd_msg.name) > 0:
            self.joint_cmd_publisher.publish(joint_cmd_msg)
    
    def calculate_control_effort(self, control_output):
        """Calculate a measure of control effort"""
        effort = 0.0
        for value in control_output.values():
            effort += abs(value)
        return effort
    
    def store_training_data(self, sensor_data, control_output):
        """Store data for future training"""
        self.ai_agent.add_training_data(sensor_data, control_output)
    
    def train_model_if_needed(self):
        """Periodically attempt to train the model if enough data is available"""
        success = self.ai_agent.train_model()
        if success:
            self.get_logger().info('AI model retrained successfully')
        else:
            self.get_logger().debug('Not enough data to retrain AI model')

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedAIBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-World Considerations

### Performance Optimization

When bridging Python AI agents with ROS 2 controllers, performance is critical:

```python
import threading
from queue import Queue, Empty
import numpy as np

class OptimizedAIBridgeNode(Node):
    def __init__(self):
        super().__init__('optimized_ai_bridge_node')
        
        # Publishers and subscribers (similar to previous example)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_publisher = self.create_publisher(JointState, '/joint_commands', 10)
        
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Separate thread for AI processing
        self.ai_input_queue = Queue(maxsize=10)  # Limit queue size
        self.ai_output_queue = Queue(maxsize=10)
        
        # Start AI processing thread
        self.ai_thread = threading.Thread(target=self.ai_processing_loop, daemon=True)
        self.ai_thread.start()
        
        # Timer for sensor data processing
        self.sensor_timer = self.create_timer(0.02, self.process_sensor_data)  # 50 Hz
        self.get_logger().info('Optimized AI Bridge Node initialized')
    
    def joint_state_callback(self, msg):
        """Non-blocking sensor data processing"""
        try:
            # Convert to numpy array for efficient processing
            joint_pos = np.array(msg.position)
            joint_vel = np.array(msg.velocity)
            
            # Prepare sensor data packet
            sensor_data = {
                'timestamp': time.time(),
                'joint_positions': joint_pos,
                'joint_velocities': joint_vel,
                'joint_names': msg.name
            }
            
            # Add to AI input queue if there's space
            try:
                self.ai_input_queue.put_nowait(sensor_data)
            except:
                # Queue is full, drop the oldest data
                try:
                    self.ai_input_queue.get_nowait()
                    self.ai_input_queue.put_nowait(sensor_data)
                except:
                    pass  # Still full, drop this data point
        except Exception as e:
            self.get_logger().error(f'Error in joint callback: {str(e)}')
    
    def ai_processing_loop(self):
        """Dedicated thread for AI processing"""
        while rclpy.ok():
            try:
                # Get the most recent sensor data
                sensor_data = None
                while True:
                    try:
                        sensor_data = self.ai_input_queue.get_nowait()
                    except Empty:
                        break  # No more recent data
                
                if sensor_data is not None:
                    # Process with AI model (this can be computationally expensive)
                    control_output = self.process_with_ai(sensor_data)
                    
                    # Add to output queue
                    try:
                        self.ai_output_queue.put_nowait(control_output)
                    except:
                        # Output queue full, drop the result
                        pass
                        
            except Exception as e:
                self.get_logger().error(f'Error in AI thread: {str(e)}')
            
            # Brief sleep to prevent busy waiting
            time.sleep(0.001)
    
    def process_with_ai(self, sensor_data):
        """AI processing function (runs in separate thread)"""
        # This is where the heavy AI computation happens
        # For example, running a neural network
        pass
    
    def process_sensor_data(self):
        """Process sensor data and send commands"""
        try:
            # Get the most recent AI output
            ai_output = None
            while True:
                try:
                    ai_output = self.ai_output_queue.get_nowait()
                except Empty:
                    break  # No more recent outputs
            
            if ai_output is not None:
                # Execute the commands produced by the AI
                self.execute_control_commands(ai_output)
        except Exception as e:
            self.get_logger().error(f'Error in sensor timer: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedAIBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Error Handling and Recovery

Robust AI-ROS bridges must handle errors gracefully:

```python
import traceback
from rclpy.qos import QoSProfile, ReliabilityPolicy

class RobustAIBridgeNode(Node):
    def __init__(self):
        super().__init__('robust_ai_bridge_node')
        
        # Publishers with custom QoS for reliability
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        
        # Subscribers (with error handling)
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.safe_joint_state_callback,
            qos_profile
        )
        
        # State tracking
        self.system_mode = 'normal'  # normal, degraded, emergency
        self.error_count = 0
        self.max_errors_before_recovery = 5
        
        # Recovery timer
        self.recovery_timer = self.create_timer(1.0, self.monitor_system_health)
        
        self.get_logger().info('Robust AI Bridge Node initialized')
    
    def safe_joint_state_callback(self, msg):
        """Safe callback with error handling"""
        try:
            self.joint_state_callback(msg)
        except Exception as e:
            self.error_count += 1
            error_msg = f'Joint state callback error: {str(e)}\n{traceback.format_exc()}'
            self.get_logger().error(error_msg)
            self.trigger_error_handling()
    
    def trigger_error_handling(self):
        """Handle errors appropriately"""
        if self.error_count >= self.max_errors_before_recovery:
            self.get_logger().error('Too many errors, entering emergency mode')
            self.system_mode = 'emergency'
            self.emergency_stop()
        elif self.error_count >= self.max_errors_before_recovery // 2:
            self.get_logger().warn('High error rate, entering degraded mode')
            self.system_mode = 'degraded'
    
    def emergency_stop(self):
        """Stop all robot motion"""
        # Publish zero velocity commands
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)
        
        # Reset joint commands to safe positions
        safe_joint_cmd = JointState()
        safe_joint_cmd.header.stamp = self.get_clock().now().to_msg()
        # Add safe joint positions here
        self.joint_cmd_publisher.publish(safe_joint_cmd)
    
    def monitor_system_health(self):
        """Monitor system health and attempt recovery"""
        if self.system_mode == 'emergency':
            self.get_logger().info('Attempting system recovery...')
            # Reset error count to allow recovery
            self.error_count = 0
            self.system_mode = 'normal'
            self.get_logger().info('System recovery attempted')
```

## Best Practices for AI-ROS Integration

### 1. Keep AI Processing Separate

Use separate threads or processes for AI computation to avoid blocking ROS communication.

### 2. Use Appropriate QoS Settings

For control commands, use reliable delivery. For sensor data, best-effort might be sufficient.

### 3. Implement Proper Error Handling

Always include try-catch blocks around AI processing and implement recovery strategies.

### 4. Monitor Performance

Track processing times and system load to ensure real-time performance.

### 5. Log Thoroughly

Keep detailed logs to debug issues that may arise from the AI-ROS interaction.

## Summary

Bridging Python AI agents with ROS 2 controllers is a critical capability for modern humanoid robots. This chapter covered the fundamental concepts:

- Using rclpy to create nodes that interface between AI systems and ROS 2
- Implementing proper communication patterns between AI and control systems
- Optimizing performance with threading and queuing
- Implementing error handling and recovery strategies
- Following best practices for robust integration

The bridge between AI algorithms and real robot control is where the intelligence of the robot meets the physical world. Proper design of this interface is critical for safe, reliable, and effective humanoid robot operation.

## Exercises

1. Create a simple bridge node that reads joint states and computes a simple control policy
2. Implement error handling in your bridge to handle sensor failures gracefully
3. Add a feedback loop that adjusts AI behavior based on control performance

## Next Steps

In the next chapter, we'll explore how to model humanoid robots using URDF (Unified Robot Description Format), which is essential for simulation and visualization in ROS 2. We'll see how the joint structures we've been discussing connect to the physical model of the robot.