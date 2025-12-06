---
sidebar_position: 2
---

# Chapter 1: Introduction to ROS 2 and Middleware Concepts

## Learning Objectives

- Understand what ROS 2 is and why it's important for robotics
- Learn about the architecture of ROS 2 and its core components
- Explore the concept of distributed systems in robotics
- Understand the role of middleware in robotics

## What is ROS 2?

ROS 2 (Robot Operating System 2) is not an operating system in the traditional sense, but rather a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Characteristics of ROS 2

1. **Distributed Architecture**: ROS 2 allows multiple processes (potentially on different machines) to communicate with each other using a publish/subscribe pattern or request/response pattern.

2. **Language Independence**: While ROS 2 is primarily developed in C++ and Python, it supports multiple programming languages through client libraries (rcl).

3. **Middleware**: ROS 2 uses DDS (Data Distribution Service) as its default middleware, which provides a vendor-neutral standard for real-time, scalable, dependable, and high-performance data exchange.

4. **Real-time Support**: Unlike ROS 1, ROS 2 has support for real-time systems.

## Core Concepts

### Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 program. A single system might have many nodes running at once, each performing a specific task.

### Packages

Packages are the software containers in ROS 2. They contain libraries, executables, scripts, or other files required for a specific functionality.

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data packets sent from publisher nodes to subscriber nodes over topics.

## ROS 2 vs ROS 1

ROS 2 was designed to address several limitations of ROS 1:

- **Real-time support**: ROS 2 supports real-time systems
- **Multi-robot support**: Better support for multi-robot systems
- **Distributed system**: No need for a master node, making it more robust
- **Security**: Built-in security features for industrial environments
- **Middleware flexibility**: Ability to switch between different middleware implementations

## Setting Up a ROS 2 Environment

To work with ROS 2, you typically need to:

1. Install ROS 2 (Humble Hawksbill is the current LTS version)
2. Source the ROS 2 setup script
3. Create a workspace for your projects
4. Create packages within that workspace

```bash
# Source the ROS 2 setup (this is typically done in your .bashrc)
source /opt/ros/humble/setup.bash

# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build
source install/setup.bash
```

## Middleware Concept in Robotics

Middleware in robotics refers to the communication layer that enables different software components to exchange information. It handles:

- Message serialization and deserialization
- Network communication protocols
- Message routing and delivery
- Quality of Service (QoS) policies
- Discovery and connection management

## Summary

ROS 2 provides the foundational middleware that connects all components of a robotic system. Its distributed architecture allows for modular development of complex robot behaviors, making it essential for humanoid robotics development.

## Exercises

1. Install ROS 2 Humble Hawksbill on your development machine
2. Create a simple ROS 2 workspace
3. Identify three key differences between ROS 1 and ROS 2

## Next Steps

In the next chapter, we'll dive deeper into ROS 2 communication patterns by exploring nodes, topics, and services.