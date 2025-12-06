---
sidebar_position: 2
---

# Chapter 1: Voice-to-Action with Whisper

## Learning Objectives

- Understand how speech recognition systems work in robotics
- Learn about OpenAI Whisper and its capabilities
- Implement a voice command pipeline for robot control
- Integrate speech recognition with action execution
- Create a complete voice-to-action system

## Introduction to Voice-to-Action Systems

Voice-to-action systems enable natural human-robot interaction by allowing users to control robots using spoken commands. These systems are particularly important for humanoid robots, as they enhance the natural interaction between humans and robotic systems.

### Key Components of Voice-to-Action Systems

1. **Speech Recognition**: Convert spoken language to text
2. **Natural Language Understanding**: Interpret the meaning of the text
3. **Action Mapping**: Map understood commands to robot actions
4. **Execution**: Perform the requested robot actions
5. **Feedback**: Provide confirmation of actions to the user

## OpenAI Whisper for Speech Recognition

OpenAI Whisper is a state-of-the-art speech recognition model that:

- Supports multiple languages
- Has robust performance across different accents and background noise
- Can be fine-tuned for specific applications
- Performs well with limited training data

### Whisper Architecture

Whisper is a transformer-based model that:

- Uses an encoder-decoder architecture
- Processes audio in 30-second chunks
- Outputs text in the detected language
- Can be prompted to focus on specific domains

### Whisper in Robotics Context

For robotics applications, Whisper can be used to:

- Convert voice commands to text that can be processed by NLP systems
- Handle background noise common in robot environments
- Support multiple languages for international applications
- Operate in real-time with appropriate computational resources

## Implementing Voice-to-Action Pipeline

The complete voice-to-action pipeline consists of:

```
[Microphone] → [Audio Preprocessing] → [Whisper ASR] → [NLU] → [Action Mapping] → [Robot Execution]
```

### Audio Preprocessing

Before sending audio to Whisper, preprocessing may include:

```python
import pyaudio
import numpy as np
import webrtcvad
from scipy.io import wavfile

# Initialize audio stream
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,  # Whisper expects 16kHz
    input=True,
    frames_per_buffer=1024
)

# Voice activity detection to identify speech segments
vad = webrtcvad.Vad()
vad.set_mode(1)  # Aggressiveness mode

# Process audio in chunks
frames = []
for i in range(0, int(16000 / 1024 * 5)):  # 5 seconds of audio
    data = stream.read(1024)
    frames.append(data)
    # Check for voice activity if needed
```

### Integrating Whisper

```python
import whisper

# Load model (use 'base' or 'small' for real-time applications)
model = whisper.load_model("base")

# Transcribe audio
result = model.transcribe("audio_file.wav")
command_text = result["text"]
print(f"Recognized command: {command_text}")
```

## Natural Language Understanding

Once speech is converted to text, we need to understand the intent:

### Simple Command Recognition

```python
# Define command patterns
COMMAND_PATTERNS = {
    "move_forward": ["move forward", "go forward", "walk forward"],
    "turn_left": ["turn left", "left turn", "rotate left"],
    "turn_right": ["turn right", "right turn", "rotate right"],
    "stop": ["stop", "halt", "freeze"],
    "wave": ["wave", "waving", "wave hello"],
    "dance": ["dance", "dancing", "perform dance"]
}

def extract_command(text):
    text_lower = text.lower()
    for action, patterns in COMMAND_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return action
    return None
```

### Using LLMs for Understanding

For more complex commands, we can use large language models:

```python
import openai

def parse_complex_command(text):
    prompt = f"""
    Parse the following human command to a robot and return the appropriate action(s):
    
    Command: "{text}"
    
    Available actions: move_forward, turn_left, turn_right, stop, wave, dance, pickup_object, place_object, speak_text, navigate_to, follow_person
    
    Response format: 
    - action: <action_name>
    - parameters: <dict with any needed parameters>
    
    If the command cannot be parsed, respond with:
    - action: "unknown"
    - parameters: {{"text": "<original command>"}}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

## Voice Command System Architecture

### Complete System Implementation

```python
import asyncio
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class VoiceCommand:
    text: str
    timestamp: float
    confidence: float

class VoiceCommandSystem:
    def __init__(self, ros_node):
        self.ros_node = ros_node
        self.command_queue = queue.Queue()
        self.is_listening = False
        self.whisper_model = whisper.load_model("base")
        
    def start_listening(self):
        self.is_listening = True
        # Start audio capture thread
        audio_thread = threading.Thread(target=self._capture_audio)
        audio_thread.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._process_commands)
        processing_thread.start()
        
    def _capture_audio(self):
        # Implementation for audio capture would go here
        pass
        
    def _process_commands(self):
        while self.is_listening:
            if not self.command_queue.empty():
                command = self.command_queue.get()
                self._execute_robot_command(command)
                
    def _execute_robot_command(self, command: VoiceCommand):
        # Map command to robot action
        action = extract_command(command.text)
        
        if action == "move_forward":
            self.ros_node.move_robot_forward()
        elif action == "turn_left":
            self.ros_node.turn_robot_left()
        elif action == "wave":
            self.ros_node.perform_wave_action()
        # ... additional mappings
```

## Integration with ROS 2

To integrate with ROS 2, we need to connect the voice system to ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control_node')
        
        # Publisher for robot movement commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Publisher for voice feedback
        self.voice_feedback_publisher = self.create_publisher(String, 'voice_feedback', 10)
        
        # Initialize voice command system
        self.voice_system = VoiceCommandSystem(self)
        
    def move_robot_forward(self):
        twist = Twist()
        twist.linear.x = 0.5  # Move forward at 0.5 m/s
        self.cmd_vel_publisher.publish(twist)
        
    def turn_robot_left(self):
        twist = Twist()
        twist.angular.z = 0.5  # Turn left at 0.5 rad/s
        self.cmd_vel_publisher.publish(twist)
        
    def perform_wave_action(self):
        # Publish to robot's action server
        # Implementation would depend on specific robot capabilities
        feedback_msg = String()
        feedback_msg.data = "Performing wave action"
        self.voice_feedback_publisher.publish(feedback_msg)
```

## Challenges in Voice-to-Action Systems

### Noise and Environment

- Background noise can affect recognition accuracy
- Robot's own sounds may interfere with recognition
- Room acoustics affect audio quality

### Language and Command Complexity

- Natural language varies greatly in how commands are expressed
- Intent recognition requires robust NLU systems
- Ambiguous commands need clarification

### Real-time Requirements

- Processing delay affects user experience
- Robot response time should match human expectations
- System should handle interruptions gracefully

## Summary

Voice-to-action systems provide a natural interface for human-robot interaction, making robots more accessible and intuitive to control. Implementing these systems requires integrating speech recognition, natural language understanding, and robot action execution.

## Exercises

1. Set up a basic audio capture system in Python
2. Install and run Whisper for speech recognition
3. Create a simple command mapping system

## Next Steps

In the next chapter, we'll explore cognitive planning systems that use Large Language Models (LLMs) to decompose complex tasks into executable subtasks.