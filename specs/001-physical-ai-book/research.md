# Research: Physical AI & Humanoid Robotics Course Book

## Decision: Simulation Platform Choice
**Rationale**: After comparing Gazebo and Unity for robotics simulation, we've decided to use both to provide a comprehensive learning experience. Gazebo will serve as the primary physics simulation environment due to its tight integration with ROS 2 and open-source nature, making it ideal for educational purposes. Unity will be used for high-fidelity rendering and human-robot interaction visualization, as it excels in creating visually rich environments.

**Alternatives considered**: 
- Gazebo only: Would limit visualization capabilities
- Unity only: Would require additional middleware for ROS integration
- Webots: Less industry-standard compared to Gazebo

## Decision: Python Agent to ROS Bridge Implementation
**Rationale**: Using rclpy (Python ROS Client Library) to bridge Python AI agents with ROS controllers. This approach provides a clean, well-documented interface that's accessible to students with Python backgrounds. The rclpy library is officially supported and maintained by the ROS community.

**Alternatives considered**:
- rospy (older ROS 1 Python library): Not compatible with ROS 2
- C++ with Boost.Python: More complex for educational purposes
- ROS Bridge with WebSockets: Additional complexity and latency

## Decision: Isaac ROS Path Planning Algorithm Selection
**Rationale**: Nav2 stack for path planning in the Isaac module. Nav2 is the standard for ROS 2 navigation, providing robust path planning capabilities including global and local planners, costmaps, and recovery behaviors. For bipedal robots, we'll focus on adapting Nav2 for legged locomotion specifically.

**Alternatives considered**:
- Custom path planning: Would be too complex for educational purposes
- MoveIt: More focused on manipulation than navigation for humanoid robots
- Other navigation libraries: Less integrated with ROS 2 ecosystem

## Decision: Chatbot Embedding Location and Retrieval Strategy
**Rationale**: Implementing the RAG chatbot using OpenAI API, FastAPI backend, Neon Postgres for conversation logs, and Qdrant for vector embeddings. This technology stack provides a scalable, cloud-ready solution that can be embedded directly within the Docusaurus-based book. The RAG (Retrieval Augmented Generation) approach will ensure answers are grounded strictly in book content.

**Alternatives considered**:
- Embedding-only model: Less flexible for complex queries
- Traditional search engines: Less contextual understanding
- On-premise solutions: Higher complexity for deployment

## Decision: Chapter-Level Depth vs Beginner Accessibility
**Rationale**: Strike a balance between providing sufficient technical depth for advanced learners while maintaining accessibility for beginners. Each chapter will follow a "pyramid approach": start with fundamental concepts, gradually introduce complexity, and include "deep-dive" sections for advanced learners. Code examples will have both simplified and complete versions.

**Alternatives considered**:
- Shallow approach: Would not satisfy advanced learners
- Advanced focus: Would alienate beginner audience
- Separate tracks: Would increase content maintenance burden

## Additional Research Findings

### Docusaurus Deployment for GitHub Pages
- Docusaurus is well-suited for documentation with built-in features for versioning, search, and navigation
- GitHub Pages deployment is straightforward with GitHub Actions
- Custom plugins can be added for the RAG chatbot integration

### ROS 2 with Humble Hawksbill
- LTS (Long Term Support) version with 5-year support
- Wide hardware support and active community
- Good for educational purposes with extensive documentation

### Isaac Sim Integration
- NVIDIA Isaac Sim provides synthetic data generation capabilities
- Isaac ROS packages provide ROS 2 interfaces for Isaac Sim
- Good for Vision-Language-Action (VLA) pipeline development

### Humanoid Robot Models
- Support for standard humanoid models like HRP-2 and NAO
- URDF (Unified Robot Description Format) for robot modeling
- Compatibility with Gazebo physics engine