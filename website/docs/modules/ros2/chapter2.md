---
sidebar_position: 3
---

# Chapter 2: ROS 2 Nodes, Topics, and Services

## Learning Objectives

- Understand the fundamental communication patterns in ROS 2
- Create and implement ROS 2 nodes for specific functionality
- Master the publish/subscribe communication model using topics
- Implement request/response communication using services
- Apply communication patterns to humanoid robot systems

## Nodes in ROS 2

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 program. A single system might have many nodes running at once, each performing a specific task.

### Creating a Node

In Python, a node is created by extending the `Node` class from `rclpy`:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle that includes:
- **Unconfigured**: Initial state after creation
- **Inactive**: Configured but not active
- **Active**: Fully operational and running
- **Finalized**: Cleanup phase before deletion

## Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data packets sent from publisher nodes to subscriber nodes over topics. The publish/subscribe paradigm is a core communication pattern in ROS 2.

### Message Types

ROS 2 provides a rich set of standard message types in the `std_msgs` package:
- `String`: Text data
- `Int32`, `Float32`: Numeric data
- `Bool`: Boolean values
- `Header`: Timestamp and frame information

Additionally, there are more specialized message types in packages like `sensor_msgs`, `geometry_msgs`, and `nav_msgs`.

### Publishing to a Topic

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()
```

### Subscribing to a Topic

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()
```

## Services in ROS 2

Services provide a request/reply communication pattern in ROS 2. A service client sends a request to a service server, which processes the request and returns a response.

### Service Types

Service types are defined using `.srv` files, which specify the request and response messages:

```
# Request message
string name
int32 age
---
# Response message
bool success
string message
```

### Creating a Service Server

```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main():
    rclpy.init()
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
```

### Creating a Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (1, 2, response.sum))
    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Quality of Service (QoS) in ROS 2

QoS profiles allow you to configure how messages are delivered between publishers and subscribers. This is important for real-time systems and reliable communication:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# Create a QoS profile for real-time performance
qos_profile = QoSProfile(
    depth=10,
    history=QoSHistoryPolicy.RMW_QOS_HISTORY_POLICY_KEEP_LAST,
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE
)

publisher = self.create_publisher(String, 'topic', qos_profile)
```

## Application to Humanoid Robotics

In humanoid robotics, ROS 2 communication patterns are used extensively:

### Joint Control

- **Topics**: Joint states published at high frequency
- **Services**: Calibration routines, mode switching
- **Actions**: Complex movements that take time to complete

### Sensor Integration

- **Topics**: Camera images, IMU data, force/torque sensors
- **Services**: Sensor configuration, calibration
- **Actions**: Long-running sensor tasks like mapping

### Navigation

- **Topics**: Odometry, laser scans, costmaps
- **Services**: Global planning, costmap updates
- **Actions**: Path following, navigation goals

## ROS 2 Tools for Communication

### ros2 topic

```bash
# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /chatter std_msgs/msg/String

# Publish a message to a topic
ros2 topic pub /chatter std_msgs/msg/String "data: Hello"
```

### ros2 service

```bash
# List all services
ros2 service list

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

## Example: Simple Humanoid Control Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class HumanoidController(Node):

    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Publishers
        self.joint_cmd_publisher = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Command subscriber
        self.command_subscriber = self.create_subscription(
            String,
            '/humanoid_commands',
            self.command_callback,
            10
        )
        
        # Store current joint states
        self.current_joint_states = JointState()
        
        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        self.current_joint_states = msg

    def command_callback(self, msg):
        command = msg.data
        self.get_logger().info(f'Received command: {command}')
        
        if command == 'wave':
            self.execute_wave_motion()
        elif command == 'stand':
            self.move_to_standing_position()

    def execute_wave_motion(self):
        # Create a joint trajectory for waving
        trajectory = JointTrajectory()
        trajectory.joint_names = ['right_shoulder_roll', 'right_elbow_pitch']
        
        # Create trajectory points
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0]  # neutral position
        point1.time_from_start.sec = 1
        trajectory.points.append(point1)
        
        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, -0.5]  # wave position
        point2.time_from_start.sec = 2
        trajectory.points.append(point2)
        
        point3 = JointTrajectoryPoint()
        point3.positions = [0.0, 0.0]  # return to neutral
        point3.time_from_start.sec = 3
        trajectory.points.append(point3)
        
        # Publish the trajectory
        self.joint_cmd_publisher.publish(trajectory)

    def move_to_standing_position(self):
        # Move to a predefined standing position
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'left_hip_pitch', 'left_knee_pitch', 'left_ankle_pitch',
            'right_hip_pitch', 'right_knee_pitch', 'right_ankle_pitch'
        ]
        
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # standing position
        point.time_from_start.sec = 2
        
        trajectory.points.append(point)
        self.joint_cmd_publisher.publish(trajectory)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Nodes, topics, and services form the foundation of communication in ROS 2 systems. Understanding these concepts is essential for developing complex robotic systems like humanoid robots. The publish/subscribe model is ideal for sensor data and state updates, while services are perfect for request/response interactions. Proper use of Quality of Service settings ensures reliable communication for time-critical applications.

## Exercises

1. Create a publisher node that publishes joint commands at 50Hz
2. Create a subscriber node that listens to IMU data and logs orientation
3. Implement a service that takes a target position and plans a trajectory

## Next Steps

In the next chapter, we'll explore how to bridge Python AI agents with ROS controllers using rclpy, connecting the AI systems we'll develop with the physical robot control.