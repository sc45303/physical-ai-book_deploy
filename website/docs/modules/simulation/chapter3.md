---
sidebar_position: 4
---

# Chapter 3: High-Fidelity Rendering & Human-Robot Interaction

## Learning Objectives

- Understand the differences between Gazebo and Unity for simulation
- Learn how Unity provides high-fidelity rendering for human-robot interaction
- Explore Unity's capabilities for creating realistic humanoid robot simulations
- Implement human-robot interaction scenarios in Unity
- Compare and contrast Gazebo and Unity for different use cases

## Introduction to High-Fidelity Rendering

High-fidelity rendering refers to the realistic visualization of environments, objects, and robots that closely resembles real-world appearance. This is important for:

1. **Human-Robot Interaction**: More realistic visual feedback improves the naturalness of human-robot interactions
2. **Training AI Models**: Realistic rendering helps AI models trained in simulation transfer better to reality
3. **User Experience**: More realistic simulations enhance the experience for users interacting with virtual robots
4. **Validation**: Realistic rendering allows for better validation of perception algorithms

## Unity vs. Gazebo: A Comparative Analysis

### Gazebo
- **Strengths**: Excellent physics simulation, ROS integration, established in robotics
- **Limitations**: Limited visual rendering capabilities, less suitable for high-fidelity visualization
- **Best for**: Physics-accurate simulation, sensor simulation, navigation and manipulation testing

### Unity
- **Strengths**: State-of-the-art rendering, game engine capabilities, realistic visual effects
- **Limitations**: Less established in robotics ecosystem, requires additional integration
- **Best for**: High-fidelity visualization, human-robot interaction studies, realistic scene generation

## Unity Robotics Integration

Unity has developed specialized tools for robotics simulation:

### Unity ML-Agents Toolkit
- **Purpose**: Reinforcement learning framework for Unity environments
- **Use case**: Training humanoid locomotion and manipulation policies
- **Integration**: Can work with ROS 2 through ROS-TCP-Connector

### Unity ROS-TCP-Connector
- **Purpose**: Enables communication between Unity and ROS 2
- **Function**: Publishes and subscribes to ROS 2 topics from within Unity
- **Use case**: Realistic visualization of complex robot behaviors

### Unity Perception Package
- **Purpose**: Generate labeled synthetic data for computer vision AI
- **Features**: Segmentation, bounding boxes, depth maps with ground truth
- **Use case**: Training perception systems with photorealistic data

## Setting Up Unity for Robotics

### Unity Robotics Hub

To integrate Unity with ROS 2, you would typically:

1. **Install Unity 2021.3 LTS or later**
2. **Install Unity Robotics Hub**
3. **Add the ROS-TCP-Connector package**
4. **Install ROS 2 Bridge tools**

### Basic Unity-ROS Integration Example

```csharp
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotTopicName = "unity_robot_command";

    // Robot properties
    public float moveSpeed = 5.0f;
    public float rotateSpeed = 100.0f;

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.instance;
        
        // Subscribe to the robot command topic
        ros.Subscribe<StringMsg>(robotTopicName, CommandCallback);
    }

    void CommandCallback(StringMsg cmd)
    {
        Debug.Log("Received command: " + cmd.data);
        
        // Process the command and update robot
        ProcessCommand(cmd.data);
    }

    void ProcessCommand(string cmd)
    {
        // Example: Parse movement commands
        if (cmd == "move_forward")
        {
            transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
        }
        else if (cmd == "turn_left")
        {
            transform.Rotate(Vector3.up, -rotateSpeed * Time.deltaTime);
        }
        else if (cmd == "turn_right")
        {
            transform.Rotate(Vector3.up, rotateSpeed * Time.deltaTime);
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Send robot state back to ROS
        if (Time.time % 0.1f < Time.deltaTime) // Send every 0.1 seconds
        {
            var robotState = new StringMsg
            {
                data = $"Position: {transform.position}, Rotation: {transform.rotation.eulerAngles}"
            };
            ros.Publish(robotTopicName + "_state", robotState);
        }
    }
}
```

## Creating Realistic Humanoid Models in Unity

### Key Components for Realistic Humanoids

1. **Rigging and Animation**:
   - Proper skeleton with realistic joint constraints
   - Blend trees for smooth transitions between movements
   - Inverse kinematics for realistic foot and hand placement

2. **Materials and Textures**:
   - PBR (Physically Based Rendering) materials for realistic surfaces
   - High-resolution textures with normal maps
   - Proper lighting models for different materials

3. **Physics Setup**:
   - Realistic physical properties (mass, friction, bounciness)
   - Proper collision shapes for accurate physics responses
   - Ragdoll physics for emergency scenarios

### Creating a Humanoid Character Controller

```csharp
using UnityEngine;

[RequireComponent(typeof(CharacterController))]
public class UnityHumanoidController : MonoBehaviour
{
    CharacterController controller;
    Animator animator;
    
    // Movement parameters
    public float walkSpeed = 2.0f;
    public float runSpeed = 4.0f;
    public float turnSpeed = 100.0f;
    public float gravity = -9.81f;
    
    // State variables
    Vector3 velocity;
    bool isGrounded;
    float speed;
    
    void Start()
    {
        controller = GetComponent<CharacterController>();
        animator = GetComponent<Animator>();
    }
    
    void Update()
    {
        // Check if character is grounded
        isGrounded = controller.isGrounded;
        if (isGrounded && velocity.y < 0)
        {
            velocity.y = -2f; // Small offset to keep grounded
        }
        
        // Handle movement input
        HandleMovement();
        
        // Apply gravity
        velocity.y += gravity * Time.deltaTime;
        
        // Move the controller
        Vector3 move = transform.right * speed + transform.up * velocity.y;
        controller.Move(move * Time.deltaTime);
        
        // Update animator parameters
        UpdateAnimator();
    }
    
    void HandleMovement()
    {
        // Get input (in a real system, this might come from ROS)
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        
        // Calculate movement direction
        Vector3 direction = new Vector3(horizontal, 0, vertical).normalized;
        
        // Calculate speed based on input
        if (direction.magnitude >= 0.1f)
        {
            float targetAngle = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg + Camera.main.transform.eulerAngles.y;
            float angle = Mathf.SmoothDampAngle(transform.eulerAngles.y, targetAngle, ref turnSpeed, 0.1f);
            transform.rotation = Quaternion.Euler(0f, angle, 0f);
            
            Vector3 moveDir = Quaternion.Euler(0f, targetAngle, 0f) * Vector3.forward;
            controller.Move(moveDir.normalized * walkSpeed * Time.deltaTime);
            
            speed = walkSpeed;
        }
        else
        {
            speed = 0f;
        }
    }
    
    void UpdateAnimator()
    {
        if (animator != null)
        {
            animator.SetFloat("Speed", speed);
            animator.SetFloat("Direction", 0); // Simplified
            animator.SetBool("IsGrounded", isGrounded);
        }
    }
}
```

## Human-Robot Interaction in Unity

### Creating Interactive Environments

```csharp
using UnityEngine;

public class InteractiveObject : MonoBehaviour
{
    public string objectName;
    public bool canBeGrabbed = true;
    
    void OnMouseOver()
    {
        // Change appearance when hovered
        GetComponent<Renderer>().material.color = Color.yellow;
    }
    
    void OnMouseExit()
    {
        // Reset appearance when not hovered
        GetComponent<Renderer>().material.color = Color.white;
    }
    
    void OnMouseDown()
    {
        // Handle interaction
        Debug.Log($"Object {objectName} was clicked");
        HandleInteraction();
    }
    
    void HandleInteraction()
    {
        // In a real system, this would send a message to ROS
        // For example, to pick up this object
        Debug.Log($"Attempting to interact with {objectName}");
    }
}

public class UnityHumanRobotInteraction : MonoBehaviour
{
    public Camera mainCamera;
    public LayerMask interactionLayer;
    public float interactionDistance = 5f;
    
    void Update()
    {
        // Check for user interaction
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, interactionDistance, interactionLayer))
            {
                // Send interaction command to ROS
                SendInteractionCommand(hit.collider.name, "interact");
            }
        }
    }
    
    void SendInteractionCommand(string objectName, string command)
    {
        // In a real system, this would send a ROS message
        Debug.Log($"Interaction command: {command} with {objectName}");
    }
}
```

## Unity Scene for Humanoid Robot Simulation

```csharp
using UnityEngine;

public class UnityRobotEnvironment : MonoBehaviour
{
    public GameObject robotPrefab;
    public Transform[] spawnPoints;
    public GameObject[] furniturePrefabs;
    public Light[] lightingSetup;
    
    [Header("Environment Settings")]
    public float gravity = -9.81f;
    public PhysicMaterial floorMaterial;
    
    void Start()
    {
        // Initialize physics settings
        Physics.gravity = new Vector3(0, gravity, 0);
        
        // Set up floor material if provided
        if (floorMaterial != null)
        {
            var floorColliders = FindObjectsOfType<Collider>();
            foreach (var col in floorColliders)
            {
                if (col.CompareTag("Floor"))
                {
                    col.material = floorMaterial;
                }
            }
        }
        
        // Spawn robot at random spawn point
        if (spawnPoints.Length > 0)
        {
            Transform spawnPoint = spawnPoints[Random.Range(0, spawnPoints.Length)];
            Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
        }
        
        // Set up lighting
        SetupLighting();
        
        // Create interactive environment
        SetupInteractiveEnvironment();
    }
    
    void SetupLighting()
    {
        // Configure lighting to enhance realism
        foreach (Light light in lightingSetup)
        {
            // Adjust lighting properties for realistic rendering
            if (light.type == LightType.Directional)
            {
                // Configure sun-like lighting
                RenderSettings.ambientIntensity = 0.5f;
                RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            }
        }
        
        // Apply realistic lighting settings
        RenderSettings.fog = true;
        RenderSettings.fogColor = Color.grey;
        RenderSettings.fogDensity = 0.01f;
    }
    
    void SetupInteractiveEnvironment()
    {
        // Create a home-like environment
        CreateRoomLayout();
        AddInteractiveElements();
    }
    
    void CreateRoomLayout()
    {
        // Create furniture and obstacles
        // This would typically be done in the Unity Editor
        // But we can programmatically set up a basic layout
        foreach (Transform spawnPoint in spawnPoints)
        {
            if (Random.Range(0, 100) > 70) // 30% chance to place furniture
            {
                if (furniturePrefabs.Length > 0)
                {
                    GameObject furniture = furniturePrefabs[Random.Range(0, furniturePrefabs.Length)];
                    Instantiate(furniture, spawnPoint.position + new Vector3(2, 0, 0), Quaternion.identity);
                }
            }
        }
    }
    
    void AddInteractiveElements()
    {
        // Add interactive elements like doors, switches, etc.
        // These would have special scripts for interaction
    }
}
```

## High-Fidelity Rendering Features in Unity

### Post-Processing Effects

Unity provides advanced post-processing features that enhance visual realism:

```csharp
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;

[RequireComponent(typeof(PostProcessVolume))]
public class RenderingController : MonoBehaviour
{
    public PostProcessVolume postProcessVolume;
    private DepthOfField depthOfField;
    private MotionBlur motionBlur;
    private AmbientOcclusion ambientOcclusion;
    
    public void SetRenderingQuality(int qualityLevel)
    {
        var profile = postProcessVolume.profile;
        
        switch (qualityLevel)
        {
            case 0: // Low quality (for real-time simulation)
                if (profile.TryGetSettings(out motionBlur))
                    motionBlur.active = false;
                if (profile.TryGetSettings(out ambientOcclusion))
                    ambientOcclusion.active = false;
                break;
                
            case 1: // Medium quality
                if (profile.TryGetSettings(out motionBlur))
                {
                    motionBlur.active = true;
                    motionBlur.shutterAngle.value = 180f;
                }
                if (profile.TryGetSettings(out ambientOcclusion))
                {
                    ambientOcclusion.active = true;
                    ambientOcclusion.intensity.value = 1.0f;
                }
                break;
                
            case 2: // High quality (for visualization)
                if (profile.TryGetSettings(out motionBlur))
                {
                    motionBlur.active = true;
                    motionBlur.shutterAngle.value = 270f;
                }
                if (profile.TryGetSettings(out ambientOcclusion))
                {
                    ambientOcclusion.active = true;
                    ambientOcclusion.intensity.value = 2.0f;
                }
                // Add other high-quality effects
                break;
        }
    }
}
```

### Dynamic Lighting and Shadows

```csharp
using UnityEngine;

public class DynamicLightingController : MonoBehaviour
{
    public Light[] sceneLights;
    public AnimationCurve lightIntensityCurve;
    public float dayNightCycleDuration = 120f; // seconds
    
    private float cycleTime = 0f;
    
    void Update()
    {
        cycleTime += Time.deltaTime;
        float normalizedTime = (cycleTime % dayNightCycleDuration) / dayNightCycleDuration;
        
        // Apply day-night cycle to lights
        foreach (Light light in sceneLights)
        {
            float intensity = lightIntensityCurve.Evaluate(normalizedTime);
            light.intensity = intensity;
            
            // Adjust color temperature based on time of day
            float colorTemperature = Mathf.Lerp(4000f, 6500f, intensity);
            light.color = GetColorForTemperature(colorTemperature);
        }
    }
    
    Color GetColorForTemperature(float temperatureK)
    {
        // Simplified color temperature calculation
        temperatureK /= 100f;
        
        float r, g, b;
        
        if (temperatureK <= 66)
        {
            r = 255;
            g = temperatureK;
            g = 99.4708025861f * Mathf.Log(g) - 161.1195681661f;
        }
        else
        {
            r = temperatureK - 60;
            r = 329.698727446f * Mathf.Pow(r, -0.1332047592f);
            g = temperatureK - 60;
            g = 288.1221695283f * Mathf.Pow(g, -0.0755148492f);
        }
        
        if (temperatureK >= 66)
        {
            b = 255;
        }
        else if (temperatureK <= 19)
        {
            b = 0;
        }
        else
        {
            b = temperatureK - 10;
            b = 138.5177312231f * Mathf.Log(b) - 305.0447927307f;
        }
        
        return new Color(r / 255f, g / 255f, b / 255f);
    }
}
```

## Integrating with ROS 2: Unity Robotics Package

### Publisher Example

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class UnityRobotPublisher : MonoBehaviour
{
    ROSConnection ros;
    string robotStateTopic = "unity_robot_state";
    string robotJointsTopic = "unity_joint_states";
    
    public Transform robotRoot;
    public Transform[] jointTransforms;
    public string[] jointNames;
    
    void Start()
    {
        ros = ROSConnection.instance;
    }
    
    void Update()
    {
        if (Time.time % 0.05f < Time.deltaTime) // Send every 50ms
        {
            PublishRobotState();
            PublishJointStates();
        }
    }
    
    void PublishRobotState()
    {
        var robotState = new PointMsg
        {
            x = robotRoot.position.x,
            y = robotRoot.position.y,
            z = robotRoot.position.z
        };
        
        ros.Publish(robotStateTopic, robotState);
    }
    
    void PublishJointStates()
    {
        var jointState = new JointStateMsg
        {
            name = jointNames,
            position = new double[jointTransforms.Length],
            velocity = new double[jointTransforms.Length],
            effort = new double[jointTransforms.Length]
        };
        
        for (int i = 0; i < jointTransforms.Length; i++)
        {
            // Get joint angles (this is simplified)
            jointState.position[i] = jointTransforms[i].localEulerAngles.y;
            jointState.velocity[i] = 0.0; // Would be calculated in a real system
            jointState.effort[i] = 0.0;   // Would be calculated in a real system
        }
        
        // Set timestamps
        jointState.header = new HeaderMsg
        {
            stamp = new TimeMsg { sec = (int)Time.time, nanosec = (uint)((Time.time % 1) * 1e9) },
            frame_id = "unity_robot"
        };
        
        ros.Publish(robotJointsTopic, jointState);
    }
}
```

### Subscriber Example

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;

public class UnityRobotSubscriber : MonoBehaviour
{
    ROSConnection ros;
    string robotCmdTopic = "unity_robot_cmd";
    
    public Transform robotRoot;
    public Transform[] jointTransforms;
    public string[] jointNames;
    
    void Start()
    {
        ros = ROSConnection.instance;
        ros.Subscribe<TwistMsg>(robotCmdTopic + "/cmd_vel", CmdVelCallback);
        ros.Subscribe<JointStateMsg>(robotCmdTopic + "/joint_commands", JointCmdCallback);
    }
    
    void CmdVelCallback(TwistMsg cmd)
    {
        // Process velocity commands
        Vector3 linearVelocity = new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z);
        Vector3 angularVelocity = new Vector3((float)cmd.angular.x, (float)cmd.angular.y, (float)cmd.angular.z);
        
        // Apply the commands to the robot
        robotRoot.Translate(linearVelocity * Time.deltaTime);
        robotRoot.Rotate(angularVelocity * Mathf.Rad2Deg * Time.deltaTime);
    }
    
    void JointCmdCallback(JointStateMsg jointCmd)
    {
        // Process joint position commands
        for (int i = 0; i < jointCmd.name.Length; i++)
        {
            string jointName = jointCmd.name[i];
            double position = jointCmd.position[i];
            
            // Find the corresponding joint transform
            for (int j = 0; j < jointNames.Length; j++)
            {
                if (jointNames[j] == jointName && j < jointTransforms.Length)
                {
                    // Apply position command to joint
                    // This is simplified - in reality, you'd use inverse kinematics
                    jointTransforms[j].localEulerAngles = new Vector3(
                        jointTransforms[j].localEulerAngles.x,
                        (float)position * Mathf.Rad2Deg,
                        jointTransforms[j].localEulerAngles.z
                    );
                    break;
                }
            }
        }
    }
}
```

## Human-Robot Interaction Scenarios

### Social Navigation

Unity enables the creation of complex social navigation scenarios:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SocialInteractionController : MonoBehaviour
{
    public List<GameObject> humanAvatars; // Humanoid characters in the scene
    public float personalSpaceRadius = 0.8f; // Comfortable distance for humans
    public float socialInteractionDistance = 2.0f; // Distance for interaction
    
    void Update()
    {
        // Check for human-robot interactions
        CheckSocialInteractions();
    }
    
    void CheckSocialInteractions()
    {
        Vector3 robotPos = transform.position;
        
        foreach (GameObject human in humanAvatars)
        {
            if (human == null) continue;
            
            Vector3 humanPos = human.transform.position;
            float distance = Vector3.Distance(robotPos, humanPos);
            
            if (distance < personalSpaceRadius)
            {
                // Too close - move away
                AvoidCollision(human);
            }
            else if (distance < socialInteractionDistance)
            {
                // In interaction range - consider greeting or yielding
                HandleSocialApproach(human);
            }
        }
    }
    
    void AvoidCollision(GameObject human)
    {
        // Calculate avoidance direction
        Vector3 direction = (transform.position - human.transform.position).normalized;
        transform.position += direction * Time.deltaTime * 0.5f;
    }
    
    void HandleSocialApproach(GameObject human)
    {
        // In a real system, this could trigger a greeting animation
        // or yield behavior
        Debug.Log("Robot is approaching human for interaction");
    }
}
```

## Realistic Sensor Simulation

### Camera Simulation with Unity Perception

```csharp
using UnityEngine;
#if UNITY_EDITOR
using Unity.Perception.GroundTruth;
#endif

public class UnityCameraSensor : MonoBehaviour
{
    public Camera sensorCamera;
    public string sensorName = "rgb_camera";
    
#if UNITY_EDITOR
    void Start()
    {
        if (sensorCamera != null)
        {
            // Add perception components for synthetic data generation
            sensorCamera.gameObject.AddComponent<SegmentationLabel>();
            sensorCamera.gameObject.AddComponent<CameraSensor>();
            
            // Configure camera for perception tasks
            var cameraSensor = sensorCamera.GetComponent<CameraSensor>();
            cameraSensor.sensorName = sensorName;
        }
    }
#endif
    
    void Update()
    {
        // In a real system, this would publish camera data to ROS
        // via the ROS-TCP-Connector
    }
}
```

## Summary

This chapter explored high-fidelity rendering for humanoid robotics using Unity, comparing it with Gazebo for different use cases:

- Unity's superior rendering capabilities for realistic visualization
- Techniques for creating realistic humanoid models in Unity
- Human-robot interaction scenarios using Unity's capabilities
- Integration of Unity with ROS 2 for robotics applications
- Advanced rendering features for realistic simulation

Unity provides powerful tools for high-fidelity rendering that complement the physics-focused simulation of Gazebo, offering a complete solution for humanoid robot development that includes realistic visualization for human-robot interaction studies.

## Exercises

1. Create a simple humanoid character in Unity with basic animations
2. Set up a basic Unity-ROS connection using ROS-TCP-Connector
3. Implement a simple human-robot interaction scenario

## Next Steps

In the next chapter, we'll explore how to integrate both Gazebo and Unity simulations in a cohesive workflow, leveraging the strengths of each platform for different aspects of humanoid robot development and testing.