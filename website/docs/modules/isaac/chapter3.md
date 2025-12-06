---
sidebar_position: 4
---

# Chapter 3: Nav2 Path Planning for Bipedal Robots

## Learning Objectives

- Understand the unique challenges of path planning for bipedal humanoid robots
- Learn Nav2 configuration for humanoid-specific navigation
- Implement dynamic path planning considering bipedal gait constraints
- Adapt global and local planners for bipedal locomotion
- Integrate footstep planning with Nav2 navigation

## Introduction to Bipedal Navigation Challenges

Navigation for humanoid robots presents unique challenges compared to wheeled robots. Bipedal locomotion requires:

1. **Dynamic Balance**: Maintaining balance while moving on two legs
2. **Footstep Planning**: Discrete footsteps rather than continuous paths
3. **ZMP (Zero Moment Point) Considerations**: Ensuring dynamic stability
4. **Terrain Adaptation**: Handling uneven surfaces, stairs, obstacles
5. **Collision Avoidance**: Avoiding obstacles while maintaining balance

### Differences from Wheeled Robot Navigation

Traditional navigation systems assume continuous, smooth motion. Bipedal robots require:

- **Discrete Path Representation**: Paths as sequences of footsteps
- **Stability Constraints**: Maintaining balance at each step
- **Dynamic Obstacle Avoidance**: Adjusting gait in real-time
- **Terrain Classification**: Different walking patterns for different surfaces

## Nav2 Architecture Overview

Nav2 consists of several key components that need to be adapted for bipedal navigation:

```
[Global Planner] → [Controller Server] → [Local Planner] → [Robot Controller]
      ↑                    ↑                   ↑
[Costmap] ← → [Costmap] ← → [Costmap]
```

For bipedal robots, these components require specific adaptations:

### Global Planner Adaptations

The global planner needs to account for:

- **Step-to-step connectivity**: Rather than continuous paths
- **Stability zones**: Areas where robot can safely place feet
- **Gait constraints**: Specific patterns for different walking speeds
- **Terrain classification**: Adapting for different surface properties

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np

class BipedalGlobalPlanner(Node):
    def __init__(self):
        super().__init__('bipedal_global_planner')
        
        # Action server for path computation
        self.action_server = self.create_action_server(
            'compute_path_to_pose',
            ComputePathToPose,
            self.execute_path_request
        )
        
        # Costmap subscription
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,  # Simplified type representation
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )
        
        self.get_logger().info('Bipedal Global Planner initialized')
    
    def execute_path_request(self, goal_handle):
        """Execute path planning request with bipedal constraints"""
        start = goal_handle.request.start
        goal = goal_handle.request.goal
        
        # Plan path considering bipedal constraints
        path = self.plan_path_with_constraints(start, goal)
        
        if path is not None:
            # Return the planned path
            result = ComputePathToPose.Result()
            result.path = path
            goal_handle.succeed()
            return result
        else:
            # Planning failed
            goal_handle.abort()
            return ComputePathToPose.Result()
    
    def plan_path_with_constraints(self, start, goal):
        """Plan path considering bipedal locomotion constraints"""
        # This is a simplified implementation
        # In reality, this would interface with footstep planners
        path = Path()
        path.header = Header()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        
        # For bipedal robots, we need to plan footstep locations
        # This is a simplified example creating a straight path
        steps = self.generate_footsteps(start.pose, goal.pose)
        
        for step in steps:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose = step
            path.poses.append(pose_stamped)
        
        return path
    
    def generate_footsteps(self, start_pose, goal_pose):
        """Generate discrete footsteps for bipedal navigation"""
        # Calculate straight line path
        dx = goal_pose.position.x - start_pose.position.x
        dy = goal_pose.position.y - start_pose.position.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Determine step size based on robot parameters
        step_length = 0.3  # meters - typical for humanoid
        step_width = 0.2   # meters - step width for balance
        
        # Generate footsteps
        footsteps = []
        num_steps = max(1, int(distance / step_length))
        
        for i in range(num_steps + 1):
            ratio = i / num_steps if num_steps > 0 else 0
            
            step_pose = Pose()
            step_pose.position.x = start_pose.position.x + dx * ratio
            step_pose.position.y = start_pose.position.y + dy * ratio
            step_pose.position.z = start_pose.position.z  # Maintain height
            
            # Add small offset for alternate feet
            if i % 2 == 0:  # Left foot
                step_pose.position.y += step_width / 2
            else:  # Right foot
                step_pose.position.y -= step_width / 2
            
            # Orientation - face toward goal
            yaw = np.arctan2(dy, dx)
            step_pose.orientation = self.yaw_to_quaternion(yaw)
            
            footsteps.append(step_pose)
        
        return footsteps
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, yaw)
        from geometry_msgs.msg import Quaternion
        quat_msg = Quaternion()
        quat_msg.x = q[0]
        quat_msg.y = q[1]
        quat_msg.z = q[2]
        quat_msg.w = q[3]
        return quat_msg
```

## Local Planner for Bipedal Robots

The local planner for bipedal robots must:

- **Generate footstep trajectories** rather than continuous velocity commands
- **Maintain dynamic balance** during obstacle avoidance
- **Adapt gait patterns** in real-time
- **Handle reactive stepping** for sudden obstacles

```python
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Path
import math

class BipedalLocalPlanner(Node):
    def __init__(self):
        super().__init__('bipedal_local_planner')
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/global_plan',
            self.global_path_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,  # Simplified type
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,  # Simplified type
            '/scan',
            self.scan_callback,
            10
        )
        
        # Publishers for footstep commands
        self.footstep_pub = self.create_publisher(
            String,  # In practice, this would be a custom footstep message
            '/footstep_commands',
            10
        )
        
        # Internal state
        self.current_path = None
        self.current_pose = None
        self.current_velocity = Twist()
        self.path_index = 0
        
        # Bipedal-specific parameters
        self.step_duration = 0.8  # seconds per step
        self.max_step_adjustment = 0.1  # meters
        self.balance_margin = 0.3  # safety margin for balance
        
        self.get_logger().info('Bipedal Local Planner initialized')
    
    def global_path_callback(self, path_msg):
        """Handle new global path"""
        self.current_path = path_msg
        self.path_index = 0
        self.get_logger().info(f'Received new path with {len(path_msg.poses)} steps')
    
    def odom_callback(self, odom_msg):
        """Update current robot pose"""
        self.current_pose = odom_msg.pose.pose
        self.current_velocity = odom_msg.twist.twist
        
        # Plan next steps based on current state
        if self.current_path:
            self.plan_next_footsteps()
    
    def scan_callback(self, scan_msg):
        """Handle laser scan for local obstacle detection"""
        # Check for obstacles in path
        min_range = min(scan_msg.ranges)
        if min_range < self.balance_margin:
            self.handle_obstacle_avoidance()
    
    def plan_next_footsteps(self):
        """Plan next few footsteps based on global path"""
        if not self.current_path or self.path_index >= len(self.current_path.poses):
            return
        
        # Get current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Determine next footsteps
        footsteps = []
        for i in range(self.path_index, min(self.path_index + 3, len(self.current_path.poses))):
            target_pose = self.current_path.poses[i].pose
            
            # Calculate required footstep with collision avoidance
            adjusted_pose = self.adjust_footstep_for_obstacles(
                current_x, current_y, 
                target_pose.position.x, target_pose.position.y
            )
            
            # Check if step is dynamically feasible
            if self.is_step_feasible(adjusted_pose):
                footsteps.append(adjusted_pose)
        
        # Publish footstep commands
        if footsteps:
            self.publish_footsteps(footsteps)
    
    def adjust_footstep_for_obstacles(self, current_x, current_y, target_x, target_y):
        """Adjust footstep to avoid obstacles"""
        # Calculate desired step
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Check for obstacles along the path
        # This is a simplified version
        step_pose = Pose()
        step_pose.position.x = target_x
        step_pose.position.y = target_y
        step_pose.position.z = 0.0  # Ground level
        
        # Add orientation
        yaw = math.atan2(dy, dx)
        step_pose.orientation = self.yaw_to_quaternion(yaw)
        
        return step_pose
    
    def is_step_feasible(self, footstep_pose):
        """Check if a footstep is dynamically feasible"""
        # Check if step is within balance limits
        # Check for obstacles at footstep location
        # Check terrain stability
        
        # Simplified feasibility check
        # In practice, this would involve ZMP calculations and terrain analysis
        return True
    
    def handle_obstacle_avoidance(self):
        """Handle local obstacle avoidance"""
        # For bipedal robots, this might involve:
        # - Adjusting next few footsteps
        # - Modifying gait parameters
        # - Temporary stopping if necessary
        
        self.get_logger().info('Obstacle detected, adjusting path')
        # Implementation would depend on specific robot and sensor setup
    
    def publish_footsteps(self, footsteps):
        """Publish planned footsteps to robot controller"""
        for i, step in enumerate(footsteps):
            step_cmd = String()
            step_cmd.data = f"STEP {self.path_index + i}: ({step.position.x:.2f}, {step.position.y:.2f})"
            self.footstep_pub.publish(step_cmd)
            
        # Update path index
        self.path_index += len(footsteps)
```

## Controller Server Adaptations

The controller server for bipedal robots needs to:

- **Convert paths to footstep sequences**
- **Generate gait patterns** based on path following requirements
- **Maintain balance** during execution
- **Handle disturbances** and recover balance

```python
class BipedalController(Node):
    def __init__(self):
        super().__init__('bipedal_controller')
        
        # Action client to send footstep commands to the robot
        self.footstep_client = ActionClient(
            self, 
            FollowFootstepPath,  # Custom action type
            'follow_footstep_path'
        )
        
        # Controller frequency
        self.controller_freq = 10  # Hz for high-level footstep planning
        self.controller_timer = self.create_timer(
            1.0 / self.controller_freq, 
            self.controller_callback
        )
        
        # Path tracking
        self.current_path = None
        self.path_progress = 0.0
        self.control_loop_counter = 0
        
        # Bipedal-specific parameters
        self.step_length = 0.3  # meters
        self.step_height = 0.02  # meters (clearance)
        self.step_duration = 0.8  # seconds per step
        
        # Balance parameters
        self.balance_threshold = 0.05  # meters max deviation
        self.recovery_enabled = True
        
        self.get_logger().info('Bipedal Controller initialized')
    
    def controller_callback(self):
        """Main control callback"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return
        
        # Determine next segment of path to follow
        next_waypoints = self.get_next_waypoints()
        
        if next_waypoints:
            # Generate footstep plan for the next segment
            footstep_plan = self.plan_footsteps(next_waypoints)
            
            # Send footstep commands to robot
            self.execute_footstep_plan(footstep_plan)
    
    def get_next_waypoints(self):
        """Get next waypoints from the global path"""
        if self.path_progress >= len(self.current_path.poses):
            return []
        
        # Return next few waypoints to plan footsteps for
        start_idx = int(self.path_progress)
        end_idx = min(start_idx + 5, len(self.current_path.poses))
        
        waypoints = []
        for i in range(start_idx, end_idx):
            waypoints.append(self.current_path.poses[i].pose)
        
        return waypoints
    
    def plan_footsteps(self, waypoints):
        """Plan detailed footsteps for a sequence of waypoints"""
        # This would interface with a footstep planner
        # For this example, we'll create a simple footstep sequence
        
        footsteps = []
        
        for i, waypoint in enumerate(waypoints):
            # Create a footstep at this location
            footstep = Footstep()  # Custom message type
            footstep.pose = waypoint
            footstep.step_type = "walk"  # Could be walk, step, turn, etc.
            footstep.duration = self.step_duration
            footstep.foot = "left" if i % 2 == 0 else "right"
            
            footsteps.append(footstep)
        
        return footsteps
    
    def execute_footstep_plan(self, footstep_plan):
        """Execute a planned sequence of footsteps"""
        if not footstep_plan:
            return
        
        # Send the plan to the robot's footstep execution system
        goal = FollowFootstepPath.Goal()
        goal.footstep_path = footstep_plan
        
        # Wait for server and send goal
        self.footstep_client.wait_for_server()
        future = self.footstep_client.send_goal_async(goal)
        
        # Handle result
        future.add_done_callback(self.footstep_execution_callback)
    
    def footstep_execution_callback(self, future):
        """Handle completion of footstep execution"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Footstep plan accepted')
        else:
            self.get_logger().error('Footstep plan rejected')
```

## Custom Costmaps for Bipedal Navigation

Bipedal robots need specialized costmaps that consider:

- **Balance zones**: Where feet can be safely placed
- **Terrain stability**: Different costs for different ground types
- **Obstacle clearance**: Sufficient space for foot placement
- **Dynamic stability**: Areas that maintain ZMP within safe limits

```python
class BipedalCostmapGenerator(Node):
    def __init__(self):
        super().__init__('bipedal_costmap_generator')
        
        # Publishers for specialized costmaps
        self.balance_costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/bipedal_balance_costmap',
            10
        )
        
        self.foot_placement_costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/bipedal_foot_placement_costmap',
            10
        )
        
        # Subscribers for sensor data
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,  # Simplified type
            '/imu',
            self.imu_callback,
            10
        )
        
        # Costmap update timer
        self.costmap_timer = self.create_timer(0.5, self.update_costmaps)
        
        # Internal data
        self.base_map = None
        self.current_terrain_type = "flat"
        self.balance_state = "stable"
        
        self.get_logger().info('Bipedal Costmap Generator initialized')
    
    def map_callback(self, map_msg):
        """Handle new map data"""
        self.base_map = map_msg
        self.get_logger().info('Received new map for costmap generation')
    
    def imu_callback(self, imu_msg):
        """Update balance state from IMU"""
        # Calculate if robot is currently in stable state
        # This is simplified - real implementation would use ZMP calculations
        orientation = imu_msg.orientation
        # Process orientation data to determine balance state
        
        # Update based on IMU data
        self.update_balance_state(imu_msg)
    
    def update_balance_state(self, imu_msg):
        """Update internal balance state based on IMU"""
        # Extract roll and pitch from orientation
        import tf_transformations
        orientation_list = [imu_msg.orientation.x, imu_msg.orientation.y, 
                           imu_msg.orientation.z, imu_msg.orientation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(orientation_list)
        
        # Determine balance state based on tilt
        tilt_magnitude = math.sqrt(roll*roll + pitch*pitch)
        if tilt_magnitude > 0.5:  # Threshold for instability
            self.balance_state = "unstable"
        else:
            self.balance_state = "stable"
    
    def update_costmaps(self):
        """Update specialized costmaps for bipedal navigation"""
        if self.base_map is None:
            return
        
        # Generate balance-based costmap
        balance_costmap = self.generate_balance_costmap()
        self.balance_costmap_pub.publish(balance_costmap)
        
        # Generate foot placement costmap
        foot_placement_costmap = self.generate_foot_placement_costmap()
        self.foot_placement_costmap_pub.publish(foot_placement_costmap)
        
        self.get_logger().debug('Costmaps updated')
    
    def generate_balance_costmap(self):
        """Generate costmap considering balance constraints"""
        # Start with base map
        costmap = OccupancyGrid()
        costmap.header = self.base_map.header
        costmap.info = self.base_map.info
        costmap.data = list(self.base_map.data)  # Copy base data
        
        # Add balance-specific costs
        for i in range(len(costmap.data)):
            if costmap.data[i] > 0:  # If there's already an obstacle cost
                continue
            
            # Calculate costs based on terrain stability, slope, etc.
            x, y = self.grid_to_world(i % costmap.info.width, i // costmap.info.width)
            cost = self.calculate_balance_cost(x, y)
            
            # Add to existing cost
            if cost > 0:
                costmap.data[i] = min(100, costmap.data[i] + int(cost * 100))
        
        return costmap
    
    def generate_foot_placement_costmap(self):
        """Generate costmap for optimal foot placement"""
        costmap = OccupancyGrid()
        costmap.header = self.base_map.header
        costmap.info = self.base_map.info
        costmap.data = list(self.base_map.data)
        
        # Consider terrain type and stability for foot placement
        for i in range(len(costmap.data)):
            if costmap.data[i] > 0:  # If obstacle
                continue
                
            x, y = self.grid_to_world(i % costmap.info.width, i // costmap.info.width)
            cost = self.calculate_foot_placement_cost(x, y)
            
            if cost > 0:
                costmap.data[i] = min(100, costmap.data[i] + int(cost * 100))
        
        return costmap
    
    def calculate_balance_cost(self, x, y):
        """Calculate balance-related cost for a position"""
        # This would implement complex balance calculations
        # For now, a simple implementation
        return 0.0  # Simplified
    
    def calculate_foot_placement_cost(self, x, y):
        """Calculate foot placement cost for a position"""
        # Consider terrain type, stability, etc.
        return 0.0  # Simplified
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = self.base_map.info.origin.position.x + grid_x * self.base_map.info.resolution
        y = self.base_map.info.origin.position.y + grid_y * self.base_map.info.resolution
        return x, y
```

## Integration with Isaac ROS Components

Isaac ROS provides GPU-accelerated perception that can enhance bipedal navigation:

```python
class IsaacBipedalIntegration(Node):
    def __init__(self):
        super().__init__('isaac_bipedal_integration')
        
        # Isaac ROS perception nodes interface
        self.depth_sub = self.create_subscription(
            Image,  # Isaac depth image
            '/isaac_ros/depth/image',
            self.depth_callback,
            10
        )
        
        self.segmentation_sub = self.create_subscription(
            Image,  # Isaac segmentation mask
            '/isaac_ros/segmentation/mask',
            self.segmentation_callback,
            10
        )
        
        # Publishers for enhanced navigation
        self.terrain_classification_pub = self.create_publisher(
            String,  # Custom message for terrain type
            '/terrain_classification',
            10
        )
        
        self.foot_placement_analysis_pub = self.create_publisher(
            String,  # Foot placement analysis results
            '/foot_placement_analysis',
            10
        )
        
        self.get_logger().info('Isaac Bipedal Integration initialized')
    
    def depth_callback(self, depth_msg):
        """Process depth data for terrain analysis"""
        # Analyze depth data to identify terrain properties
        # - Surface inclination for balance planning
        # - Obstacle detection for foot placement
        # - Step height for stair navigation
        
        # This would use Isaac's GPU-accelerated processing
        terrain_info = self.analyze_terrain_from_depth(depth_msg)
        self.terrain_classification_pub.publish(terrain_info)
    
    def segmentation_callback(self, seg_msg):
        """Process segmentation data for foot placement"""
        # Analyze segmentation to identify:
        # - Traversable surfaces vs obstacles
        # - Different terrain types (grass, concrete, etc.)
        # - Hazardous areas (water, holes, etc.)
        
        foot_placement_analysis = self.analyze_foot_placement_area(seg_msg)
        self.foot_placement_analysis_pub.publish(foot_placement_analysis)
    
    def analyze_terrain_from_depth(self, depth_msg):
        """Analyze terrain properties from depth data"""
        # GPU-accelerated terrain analysis using Isaac tools
        # This would implement complex terrain classification
        result = String()
        result.data = "terrain_analysis_result"
        return result
    
    def analyze_foot_placement_area(self, seg_msg):
        """Analyze suitable areas for foot placement"""
        # Use segmentation to identify safe foot placement zones
        result = String()
        result.data = "foot_placement_analysis_result"
        return result
```

## Footstep Planning Integration

A complete bipedal navigation system integrates high-level path planning with footstep generation:

```python
class IntegratedBipedalNavigator(Node):
    def __init__(self):
        super().__init__('integrated_bipedal_navigator')
        
        # Interface with Nav2 components
        self.path_client = ActionClient(
            self, 
            ComputePathToPose, 
            'compute_path_to_pose'
        )
        
        # Interface with footstep planner
        self.footstep_planner = ActionClient(
            self,
            PlanFootsteps,
            'plan_footsteps'
        )
        
        # Navigation command interface
        self.nav_goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.navigation_goal_callback,
            10
        )
        
        # Initialize components
        self.bipedal_costmap_generator = BipedalCostmapGenerator(self)
        self.bipedal_local_planner = BipedalLocalPlanner(self)
        
        self.get_logger().info('Integrated Bipedal Navigator initialized')
    
    def navigation_goal_callback(self, goal_msg):
        """Handle new navigation goal"""
        # Plan high-level path first
        self.plan_global_path(goal_msg.pose)
    
    def plan_global_path(self, goal_pose):
        """Plan global path considering bipedal constraints"""
        # Create path planning request
        path_goal = ComputePathToPose.Goal()
        path_goal.goal = goal_pose
        # Add bipedal-specific parameters
        path_goal.planner_id = "bipedal_global_planner"
        
        # Send to global planner
        self.path_client.wait_for_server()
        future = self.path_client.send_goal_async(path_goal)
        future.add_done_callback(self.global_path_callback)
    
    def global_path_callback(self, future):
        """Handle global path planning result"""
        goal_handle = future.result()
        if goal_handle.accepted:
            path_result = goal_handle.result()
            # Plan detailed footsteps for the path
            self.plan_detailed_footsteps(path_result.path)
        else:
            self.get_logger().error('Global path planning failed')
    
    def plan_detailed_footsteps(self, path):
        """Plan detailed footsteps for a high-level path"""
        footstep_goal = PlanFootsteps.Goal()
        footstep_goal.path = path
        footstep_goal.robot_properties = self.get_robot_properties()
        
        self.footstep_planner.wait_for_server()
        future = self.footstep_planner.send_goal_async(footstep_goal)
        future.add_done_callback(self.footstep_plan_callback)
    
    def footstep_plan_callback(self, future):
        """Handle footstep planning result"""
        goal_handle = future.result()
        if goal_handle.accepted:
            footstep_result = goal_handle.result()
            # Execute the planned footsteps
            self.execute_navigation(footstep_result.footsteps)
        else:
            self.get_logger().error('Footstep planning failed')
    
    def execute_navigation(self, footsteps):
        """Execute the planned navigation"""
        # Interface with robot's walking controller
        # Monitor execution and handle replanning if needed
        self.get_logger().info(f'Executing navigation with {len(footsteps)} steps')
    
    def get_robot_properties(self):
        """Get robot-specific properties for planning"""
        # Return properties like step length, width, height, etc.
        from builtin_interfaces.msg import Duration
        props = RobotProperties()
        props.max_step_length = 0.3  # meters
        props.max_step_width = 0.2   # meters
        props.step_height_clearance = 0.05
        props.balance_margin = 0.1   # meters
        props.zmp_stability_threshold = 0.05  # meters
        props.walk_period = Duration(sec=0, nanosec=800000000)  # 0.8 seconds
        
        return props
```

## Performance Optimization

For real-time bipedal navigation, optimization is critical:

```python
class OptimizedBipedalPlanner:
    def __init__(self):
        # Pre-computed lookup tables for common gait patterns
        self.gait_patterns = self.precompute_gait_patterns()
        
        # Cached inverse kinematics solutions
        self.ik_cache = {}
        
        # Multi-resolution path planning
        self.coarse_planner = None
        self.fine_planner = None
        
    def precompute_gait_patterns(self):
        """Pre-compute common gait patterns for fast lookup"""
        # This would contain pre-computed footstep sequences
        # for common movement patterns (forward, backward, turns, etc.)
        patterns = {}
        # Forward steps
        patterns['forward'] = self.compute_standard_gait(0.0, 0.3, 0.0)  # x, y, theta
        # Turning steps
        patterns['turn_left'] = self.compute_standard_gait(0.1, 0.0, 0.2)  # Small forward + rotation
        patterns['turn_right'] = self.compute_standard_gait(0.1, 0.0, -0.2)
        
        return patterns
    
    def compute_standard_gait(self, dx, dy, dtheta):
        """Compute a standard gait pattern for given movement"""
        # Simplified gait pattern computation
        # In practice, this would use complex biomechanical models
        footsteps = []
        # Implementation would generate appropriate footstep sequence
        return footsteps
```

## Safety and Recovery Mechanisms

Bipedal navigation systems need robust safety mechanisms:

```python
class BipedalSafetyManager(Node):
    def __init__(self):
        super().__init__('bipedal_safety_manager')
        
        # Monitor system health
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Emergency stop publisher
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        
        # Recovery action client
        self.recovery_client = ActionClient(self, ExecuteRecovery, 'execute_recovery')
        
        # Safety timers
        self.safety_timer = self.create_timer(0.1, self.check_safety)
        
        # Safety thresholds
        self.tilt_threshold = 0.785  # 45 degrees
        self.velocity_threshold = 1.0  # m/s
        self.joint_limit_threshold = 0.1  # radians from limit
        
        self.current_tilt = 0.0
        self.current_velocity = 0.0
        self.in_safe_state = True
        
        self.get_logger().info('Bipedal Safety Manager initialized')
    
    def imu_callback(self, imu_msg):
        """Monitor robot's balance state"""
        # Calculate tilt angle from IMU
        import tf_transformations
        orientation_list = [imu_msg.orientation.x, imu_msg.orientation.y, 
                           imu_msg.orientation.z, imu_msg.orientation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(orientation_list)
        self.current_tilt = math.sqrt(roll*roll + pitch*pitch)
    
    def joint_state_callback(self, joint_state_msg):
        """Monitor joint positions for safety"""
        # Check joint limits and velocities
        pass
    
    def check_safety(self):
        """Check if robot is in safe state"""
        if self.current_tilt > self.tilt_threshold:
            self.trigger_recovery("balance_loss")
        elif self.current_velocity > self.velocity_threshold:
            self.trigger_recovery("speed_violation")
    
    def trigger_recovery(self, reason):
        """Trigger recovery procedure"""
        self.get_logger().warn(f'Safety violation: {reason}, initiating recovery')
        
        # Send emergency stop
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
        
        # Trigger recovery action
        recovery_goal = ExecuteRecovery.Goal()
        recovery_goal.recovery_type = reason
        self.recovery_client.wait_for_server()
        self.recovery_client.send_goal_async(recovery_goal)
```

## Summary

This chapter covered the adaptation of Nav2 for bipedal humanoid robots:

- Unique challenges of bipedal navigation compared to wheeled robots
- Modifications to Nav2 components (global planner, local planner, controller) for bipedal locomotion
- Integration of footstep planning with traditional path planning
- Specialized costmaps for balance and foot placement
- Safety and recovery mechanisms for bipedal robots
- Performance optimization techniques

Bipedal navigation requires fundamentally different approaches than traditional mobile robotics, focusing on discrete footsteps, balance maintenance, and gait adaptation rather than continuous velocity control.

## Exercises

1. Create a simple footstep planner that generates footsteps between two points
2. Implement a costmap that considers terrain stability for foot placement
3. Design a safety mechanism that detects balance loss and triggers recovery

## Next Steps

In the next chapter, we'll explore advanced AI-robot brain techniques that leverage the perception and navigation capabilities we've developed, focusing on higher-level cognitive functions and learning for humanoid robots.