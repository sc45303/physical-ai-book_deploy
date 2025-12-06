---
sidebar_position: 3
---

# Chapter 2: Cognitive Planning using LLMs

## Learning Objectives

- Understand how Large Language Models (LLMs) can be used for cognitive planning
- Implement cognitive planning systems for humanoid robots
- Integrate LLM-based planning with action execution
- Learn about task decomposition and hierarchical planning
- Create robust planning systems that handle ambiguity and errors

## Introduction to Cognitive Planning

Cognitive planning in robotics refers to the high-level decision-making process that determines what actions a robot should take to achieve its goals. For humanoid robots, this involves understanding natural language commands, decomposing complex tasks into simpler ones, and adapting to dynamic environments.

### The Role of LLMs in Cognitive Planning

Large Language Models have revolutionized cognitive planning by:

1. **Natural Language Understanding**: Converting human commands to robot actions
2. **Task Decomposition**: Breaking complex tasks into executable sequences
3. **Context Management**: Maintaining understanding of the environment and goals
4. **Adaptation**: Adjusting plans based on new information or obstacles

### Cognitive Planning vs. Low-level Planning

- **Cognitive Planning**: High-level goal-directed behavior, task decomposition, handling ambiguous commands
- **Low-level Planning**: Path planning, trajectory generation, motion control

## Architecture of LLM-Based Cognitive Planning

The cognitive planning system typically follows this architecture:

```
[Human Command] → [NLU with LLM] → [Task Decomposition] → [Plan Refinement] → [Action Execution] → [Feedback Loop]
```

### Natural Language Understanding with LLMs

LLMs can understand complex, natural language commands:

```python
import openai

class CognitivePlanner:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key
        self.robot_capabilities = [
            "move_forward", "turn_left", "turn_right", "stop",
            "pick_up_object", "place_object", "speak_text",
            "wave_gesture", "navigate_to", "find_object"
        ]
        
    def parse_command(self, command_text):
        """Convert natural language to structured action plan"""
        
        prompt = f"""
        You are a cognitive planning system for a humanoid robot.
        Convert the following human command to a sequence of robot actions.
        Select from the following capabilities: {self.robot_capabilities}
        
        Human Command: "{command_text}"
        
        Return the plan as a JSON array of actions, where each action has:
        - action: name of the action (from the capabilities list)
        - parameters: object with required parameters for the action
        
        Example response format:
        [
            {{"action": "navigate_to", "parameters": {{"location": "kitchen"}}}},
            {{"action": "find_object", "parameters": {{"object": "bottle"}}}},
            {{"action": "pick_up_object", "parameters": {{"object": "bottle"}}}},
            {{"action": "navigate_to", "parameters": {{"location": "table"}}}},
            {{"action": "place_object", "parameters": {{"object": "bottle", "location": "table"}}}}
        ]
        
        Command:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent outputs
            max_tokens=500
        )
        
        import json
        try:
            # Extract JSON from the response
            content = response.choices[0].message.content
            # Find JSON in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                return plan
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, return a default plan
            return [{"action": "speak_text", "parameters": {"text": f"I don't understand the command: {command_text}"}}]
        
        return [{"action": "speak_text", "parameters": {"text": f"I don't understand the command: {command_text}"}}]
```

## Task Decomposition and Hierarchical Planning

Complex commands need to be broken down into simpler, executable actions:

### Example: "Bring me a drink from the kitchen"

This high-level command decomposes into:

1. **Goal**: Deliver a drink to the user
2. **Subgoals**:
   - Navigate to the kitchen
   - Identify a drink
   - Pick up the drink
   - Navigate back to the user
   - Present the drink

### Implementation of Hierarchical Planner

```python
class HierarchicalPlanner:
    def __init__(self, llm_planner):
        self.llm_planner = llm_planner
        self.known_locations = {
            "kitchen": [1.0, 2.0, 0.0],
            "living_room": [0.0, 0.0, 0.0],
            "bedroom": [-2.0, 1.0, 0.0],
            "dining_room": [-1.0, -1.0, 0.0]
        }
        
        self.known_objects = {
            "drink": ["water_bottle", "soda_can", "juice_box"],
            "snack": ["cookies", "apple", "chips"],
            "tool": ["screwdriver", "wrench", "hammer"]
        }
    
    def create_plan(self, high_level_command):
        """Create a plan for a high-level command using LLM and domain knowledge"""
        
        # First, use LLM to get a general plan structure
        general_plan = self.llm_planner.parse_command(high_level_command)
        
        # Then, refine the plan using domain knowledge
        refined_plan = self.refine_plan(general_plan, high_level_command)
        
        return refined_plan
    
    def refine_plan(self, plan, command):
        """Refine the plan using domain knowledge and context"""
        refined = []
        
        for action in plan:
            if action["action"] == "navigate_to" and "location" in action["parameters"]:
                location = action["parameters"]["location"]
                
                # Resolve location to coordinates
                if location in self.known_locations:
                    coords = self.known_locations[location]
                    refined.append({
                        "action": "navigate_to_coordinates",
                        "parameters": {"x": coords[0], "y": coords[1], "theta": coords[2]}
                    })
                else:
                    # If location not known, add a search step
                    refined.append({
                        "action": "search_for_location",
                        "parameters": {"location": location}
                    })
                    
            elif action["action"] == "find_object" and "object" in action["parameters"]:
                obj_type = action["parameters"]["object"]
                
                # Expand object type to specific objects
                if obj_type in self.known_objects:
                    possible_objects = self.known_objects[obj_type]
                    refined.append({
                        "action": "search_for_objects",
                        "parameters": {"objects": possible_objects}
                    })
                else:
                    refined.append(action)  # Keep original action
                    
            else:
                refined.append(action)
        
        return refined
```

## Context Management and Memory

Cognitive planning systems need to maintain context across multiple interactions:

### Memory Systems

```python
import datetime
from typing import List, Dict, Any

class ContextManager:
    def __init__(self):
        self.episodic_memory = []  # Recent interactions
        self.semantic_memory = {}  # General knowledge about the world
        self.procedural_memory = {}  # How to perform tasks
        
    def update_context(self, event: Dict[str, Any]):
        """Update the context with a new event"""
        event_with_timestamp = {
            "timestamp": datetime.datetime.now(),
            "event": event
        }
        self.episodic_memory.append(event_with_timestamp)
        
        # Keep only recent events (last 100)
        if len(self.episodic_memory) > 100:
            self.episodic_memory = self.episodic_memory[-100:]
    
    def get_recent_context(self, timeframe_minutes=30):
        """Get context from the last timeframe_minutes"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=timeframe_minutes)
        recent_events = [
            event for event in self.episodic_memory 
            if event["timestamp"] > cutoff_time
        ]
        return recent_events
    
    def infer_state(self):
        """Infer the current state of the world from context"""
        recent_events = self.get_recent_context()
        
        # Example: Infer robot location from navigation events
        current_location = "unknown"
        for event in reversed(recent_events):
            if event["event"]["type"] == "navigation" and event["event"]["status"] == "completed":
                current_location = event["event"]["destination"]
                break
        
        # Example: Infer task progress
        current_task = "idle"
        for event in reversed(recent_events):
            if event["event"]["type"] == "task":
                current_task = event["event"]["task_name"]
                break
        
        return {
            "location": current_location,
            "task": current_task,
            "recent_events": recent_events[-10:]  # Last 10 events
        }
```

## Handling Ambiguity and Clarification

Real-world commands are often ambiguous and need clarification:

```python
class AmbiguityResolver:
    def __init__(self, context_manager):
        self.context_manager = context_manager
        
    def resolve_ambiguity(self, command):
        """Determine if command is ambiguous and what clarification is needed"""
        context = self.context_manager.infer_state()
        
        # Example ambiguities to check
        if "it" in command.lower() or "that" in command.lower():
            # Check if "it" or "that" refers to something in context
            if not self.resolve_pronoun(command, context):
                return {
                    "ambiguous": True,
                    "clarification_needed": "What does 'it' or 'that' refer to?",
                    "options": self.get_possible_referents(context)
                }
        
        if "there" in command.lower():
            # Unclear location reference
            return {
                "ambiguous": True,
                "clarification_needed": "Where specifically do you mean?",
                "options": ["kitchen", "living room", "bedroom", "dining room"]
            }
        
        # Check for ambiguous object references
        import re
        object_patterns = re.findall(r'the (\w+)', command.lower())
        for obj in object_patterns:
            if self.is_ambiguous_object(obj, context):
                return {
                    "ambiguous": True,
                    "clarification_needed": f"Which {obj} do you mean?",
                    "options": self.get_specific_objects(obj, context)
                }
        
        return {"ambiguous": False}
    
    def resolve_pronoun(self, command, context):
        """Try to resolve pronouns like 'it' or 'that' using context"""
        # Implementation would use context to resolve pronouns
        # For simplicity, return False to trigger clarification
        return False
    
    def is_ambiguous_object(self, obj, context):
        """Check if an object reference is ambiguous"""
        # Implementation would check if multiple instances exist
        return obj in ["object", "item", "thing", "one"]
    
    def get_specific_objects(self, obj, context):
        """Get specific instances of an object type"""
        return [f"{obj}_1", f"{obj}_2", f"{obj}_3"]
    
    def get_possible_referents(self, context):
        """Get possible referents for pronouns"""
        return ["the bottle", "the chair", "the table", "the person"]
```

## Plan Execution and Monitoring

Once a plan is created, it needs to be executed and monitored:

```python
class PlanExecutor:
    def __init__(self, robot_interface, context_manager):
        self.robot_interface = robot_interface
        self.context_manager = context_manager
        self.current_plan = None
        self.current_step = 0
        
    def execute_plan(self, plan):
        """Execute a plan step by step"""
        self.current_plan = plan
        self.current_step = 0
        
        for i, action in enumerate(plan):
            self.current_step = i
            success = self.execute_action(action)
            
            if not success:
                return self.handle_failure(action, i)
        
        return True
    
    def execute_action(self, action):
        """Execute a single action"""
        action_type = action["action"]
        parameters = action.get("parameters", {})
        
        try:
            if action_type == "navigate_to_coordinates":
                return self.robot_interface.navigate_to(
                    parameters["x"], 
                    parameters["y"], 
                    parameters.get("theta", 0.0)
                )
            elif action_type == "pick_up_object":
                return self.robot_interface.pick_up_object(
                    parameters["object"]
                )
            elif action_type == "place_object":
                return self.robot_interface.place_object(
                    parameters["object"], 
                    parameters["location"]
                )
            elif action_type == "speak_text":
                return self.robot_interface.speak_text(
                    parameters["text"]
                )
            else:
                self.robot_interface.speak_text(f"Unknown action: {action_type}")
                return False
                
        except Exception as e:
            self.robot_interface.speak_text(f"Error executing action: {str(e)}")
            return False
    
    def handle_failure(self, failed_action, step_index):
        """Handle plan execution failure"""
        self.context_manager.update_context({
            "type": "failure",
            "action": failed_action,
            "step": step_index,
            "reason": "action_execution_failed"
        })
        
        # For this implementation, return False to indicate failure
        # A more sophisticated system might have recovery actions
        return False
```

## Integration with ROS 2

Integrating cognitive planning with ROS 2 requires proper message passing:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from cognitive_planning_msgs.msg import Plan, PlanStep  # Custom message

class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner_node')
        
        # Publishers
        self.plan_publisher = self.create_publisher(Plan, 'robot_plan', 10)
        self.feedback_publisher = self.create_publisher(String, 'planner_feedback', 10)
        
        # Subscribers
        self.command_subscriber = self.create_subscription(
            String,
            'high_level_command',
            self.command_callback,
            10
        )
        
        # Initialize planner components
        self.context_manager = ContextManager()
        self.ambiguity_resolver = AmbiguityResolver(self.context_manager)
        self.hierarchical_planner = HierarchicalPlanner(CognitivePlanner(openai_api_key="YOUR_KEY"))
        self.plan_executor = PlanExecutor(RobotInterface(), self.context_manager)
        
        self.get_logger().info('Cognitive Planner Node initialized')
    
    def command_callback(self, msg):
        """Process a high-level command"""
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')
        
        # Check for ambiguity
        ambiguity_check = self.ambiguity_resolver.resolve_ambiguity(command_text)
        if ambiguity_check["ambiguous"]:
            feedback_msg = String()
            feedback_msg.data = f"Clarification needed: {ambiguity_check['clarification_needed']}"
            self.feedback_publisher.publish(feedback_msg)
            return
        
        # Create and execute plan
        plan = self.hierarchical_planner.create_plan(command_text)
        
        # Publish the plan
        plan_msg = self.convert_to_ros_plan(plan)
        self.plan_publisher.publish(plan_msg)
        
        # Execute the plan
        success = self.plan_executor.execute_plan(plan)
        
        # Report results
        result_msg = String()
        if success:
            result_msg.data = f"Successfully executed command: {command_text}"
        else:
            result_msg.data = f"Failed to execute command: {command_text}"
        
        self.feedback_publisher.publish(result_msg)
        
        # Update context
        self.context_manager.update_context({
            "type": "command_execution",
            "command": command_text,
            "result": "success" if success else "failure"
        })
    
    def convert_to_ros_plan(self, plan):
        """Convert internal plan representation to ROS message"""
        plan_msg = Plan()
        
        for i, step in enumerate(plan):
            step_msg = PlanStep()
            step_msg.id = i
            step_msg.action = step["action"]
            
            # Convert parameters to string format for simplicity
            import json
            step_msg.parameters = json.dumps(step.get("parameters", {}))
            
            plan_msg.steps.append(step_msg)
        
        return plan_msg

# Example RobotInterface class (simplified)
class RobotInterface:
    def __init__(self):
        pass
    
    def navigate_to(self, x, y, theta):
        # Implementation would send navigation goals to Nav2
        print(f"Navigating to ({x}, {y}, {theta})")
        return True  # Simulated success
    
    def pick_up_object(self, obj_name):
        # Implementation would control manipulator
        print(f"Attempting to pick up {obj_name}")
        return True  # Simulated success
    
    def place_object(self, obj_name, location):
        # Implementation would control manipulator
        print(f"Placing {obj_name} at {location}")
        return True  # Simulated success
    
    def speak_text(self, text):
        # Implementation would use text-to-speech
        print(f"Robot says: {text}")
        return True  # Simulated success

def main(args=None):
    rclpy.init(args=args)
    planner_node = CognitivePlannerNode()
    
    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Error Handling and Recovery

Robust cognitive planning systems need to handle errors gracefully:

```python
class RecoverySystem:
    def __init__(self):
        self.recovery_strategies = {
            "navigation_failure": [
                "try_alternative_path",
                "request_human_help",
                "wait_and_retry"
            ],
            "object_not_found": [
                "expand_search_area",
                "ask_for_help",
                "substitute_alternative"
            ],
            "grasp_failure": [
                "adjust_grasp_approach",
                "request_assistance",
                "try_different_object"
            ]
        }
    
    def suggest_recovery(self, failure_type):
        """Suggest recovery strategies for a given failure type"""
        if failure_type in self.recovery_strategies:
            return self.recovery_strategies[failure_type]
        else:
            return ["request_human_help"]
    
    def execute_recovery(self, strategy, context):
        """Execute a recovery strategy"""
        if strategy == "try_alternative_path":
            # Implementation would involve replanning with Nav2
            print("Attempting alternative navigation path...")
            return True
        elif strategy == "request_human_help":
            # Ask human for assistance
            print("Requesting human assistance...")
            return False  # Need human intervention
        elif strategy == "expand_search_area":
            # Increase the area to search for an object
            print("Expanding search area...")
            return True
        else:
            print(f"Unknown recovery strategy: {strategy}")
            return False
```

## Summary

Cognitive planning with LLMs enables humanoid robots to understand and execute complex, natural language commands. By combining LLMs for natural language understanding and task decomposition with robust execution monitoring and error handling, we can create sophisticated robotic systems that interact naturally with humans. The key components include natural language understanding, hierarchical planning, context management, ambiguity resolution, and error recovery.

## Exercises

1. Implement a simple cognitive planner that can handle basic commands
2. Add context management to track the robot's state across interactions
3. Create a system that asks for clarification when commands are ambiguous

## Next Steps

In the next chapter, we'll explore how to integrate vision, language, and action systems into a complete pipeline, bringing together all the components learned in this course into a cohesive system for humanoid robots.