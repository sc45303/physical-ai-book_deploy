# Quickstart Guide: Physical AI & Humanoid Robotics Course Book

## Overview
This guide provides a quick overview of how to get started with developing and deploying the Physical AI & Humanoid Robotics Course Book.

## Prerequisites
- Git
- Node.js (v18 or higher)
- Python 3.9+
- ROS 2 Humble Hawksbill
- Docker (optional, for containerized development)

## Setting up the Development Environment

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/physical-ai-course.git
cd physical-ai-course
```

### 2. Install Docusaurus Dependencies
```bash
npm install
```

### 3. Set up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install ROS 2 Dependencies
Follow the official ROS 2 Humble installation guide for your platform:
- [Ubuntu Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [Windows Installation Guide](https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html)

## Running the Book Locally

### 1. Start the Docusaurus Development Server
```bash
npm start
```
This will start the book at http://localhost:3000

### 2. Running Simulations
For each chapter with simulation content:

#### ROS 2 Examples
```bash
source /opt/ros/humble/setup.bash
cd path/to/chapter/examples
python3 example_script.py
```

#### Gazebo Simulation
```bash
source /opt/ros/humble/setup.bash
ros2 launch your_package your_simulation.launch.py
```

#### Isaac Sim
1. Install Isaac Sim (requires NVIDIA GPU)
2. Follow the setup guide in the Isaac module

## RAG Chatbot Setup

### 1. Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
NEON_DB_URL=your_neon_db_connection_string
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 2. Start the Chatbot Backend
```bash
cd backend
uvicorn main:app --reload
```

### 3. Test the Chatbot API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ROS 2?", "moduleContext": "ROS 2 - The Robotic Nervous System"}'
```

## Adding Content

### 1. Creating a New Chapter
```bash
# Navigate to the appropriate module directory
cd docs/modules/ros2  # or gazebo, isaac, vla
# Create the new chapter file
touch new-chapter.md
```

### 2. Chapter Template
```md
---
title: Your Chapter Title
sidebar_position: 1
description: Brief chapter description
---

# Your Chapter Title

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Introduction
Introductory content...

## Main Content
Detailed content with:
- Explanations
- Code examples
- Diagrams
- Activities

## Summary
Chapter summary...

## Exercises
- Exercise 1
- Exercise 2
```

### 3. Adding Code Examples
```md
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

## Building for Production

### 1. Build the Static Site
```bash
npm run build
```

### 2. Build and Deploy
The site can be deployed to GitHub Pages, Netlify, or any static hosting service.

### 3. Continuous Deployment
We use GitHub Actions for automatic deployment when content is merged to the main branch.

## Validation and Testing

### 1. Validating Content
```bash
npm run validate
```

### 2. Testing Code Examples
For each module:
```bash
cd path/to/module/examples
python -m pytest tests/
```

### 3. Testing RAG Accuracy
Run a set of predefined questions against the RAG system:
```bash
python tests/test_rag_accuracy.py
```

## Module Structure
The course is organized into 4 modules, each to be completed fully before moving to the next:

1. **ROS 2 – The Robotic Nervous System** (3-4 chapters)
2. **Gazebo & Unity – The Digital Twin** (3-4 chapters) 
3. **NVIDIA Isaac – The AI-Robot Brain** (3-4 chapters)
4. **Vision-Language-Action (VLA)** (3-4 chapters)

Each module includes hands-on tutorials, code examples, simulations, and ends with a mini-project before moving to the next module.

The final capstone project integrates all concepts into "The Autonomous Humanoid".

## Getting Help
- For technical issues: Open an issue in the repository
- For content questions: Use the embedded RAG chatbot
- For feedback: Contact the course development team