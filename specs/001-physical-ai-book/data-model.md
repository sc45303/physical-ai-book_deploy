# Data Model: Physical AI & Humanoid Robotics Course Book

## Course Modules
- **Name**: String, required (e.g., "ROS 2: The Robotic Nervous System")
- **Description**: Text, required
- **ModuleNumber**: Integer, required (1-4)
- **Chapters**: List of Chapter entities
- **LearningObjectives**: List of strings, required
- **Prerequisites**: List of strings
- **Duration**: String (e.g., "4 weeks")

## Chapter
- **Title**: String, required
- **ModuleId**: Reference to CourseModules, required
- **ChapterNumber**: Integer within module, required
- **Content**: Markdown text, required
- **LearningObjectives**: List of strings, required
- **Activities**: List of Activity entities
- **CodeExamples**: List of CodeExample entities
- **Diagrams**: List of Diagram entities
- **Duration**: String (e.g., "1 week")

## Activity
- **Type**: Enum (Tutorial, Lab, Exercise, Quiz), required
- **Title**: String, required
- **Description**: Text, required
- **Instructions**: Markdown text, required
- **ExpectedOutcome**: Text, required
- **Resources**: List of strings (file paths or URLs)
- **Difficulty**: Enum (Beginner, Intermediate, Advanced)

## CodeExample
- **Title**: String, required
- **Description**: Text
- **Language**: String (e.g., "Python", "C++"), required
- **Code**: Text (raw code), required
- **Explanation**: Markdown text explaining the code
- **FilePath**: String (path in the project structure)
- **Testable**: Boolean (whether the example can be run/tested)

## Diagram
- **Title**: String, required
- **Description**: Text
- **Type**: Enum (Block, Flowchart, Architecture, Process)
- **FilePath**: String (path to diagram file)
- **AltText**: String (for accessibility)

## RAG Chatbot
- **Name**: String
- **Description**: Text
- **KnowledgeBase**: Reference to BookContent
- **QueryInterface**: String (e.g., "embedded widget")
- **ResponseFormat**: Enum (Text, Markdown, JSON)
- **ValidationMethod**: String (how hallucinations are prevented)

## Humanoid Robot Simulation
- **Name**: String, required
- **Description**: Text
- **RobotModel**: String (URDF file reference)
- **SimulationEnvironment**: String (Gazebo/Unity world file)
- **Capabilities**: List of strings (e.g., "voice commands", "navigation", "manipulation")
- **Sensors**: List of Sensor entities
- **Actuators**: List of Actuator entities

## Sensor
- **Type**: Enum (Camera, LIDAR, IMU, GPS, ForceTorque), required
- **Name**: String, required
- **SimulationModel**: String (plugin/model name)
- **Parameters**: Dictionary of sensor parameters
- **Topics**: List of ROS topic names

## Actuator
- **Type**: Enum (Joint, Gripper, Wheel), required
- **Name**: String, required
- **JointType**: Enum (Revolute, Prismatic, Fixed)
- **ControlMethod**: String (e.g., "position", "velocity", "effort")
- **Topics**: List of ROS topic names

## BookContent
- **Title**: String, required
- **Body**: Markdown text, required
- **ModuleId**: Reference to CourseModules
- **ChapterId**: Reference to Chapter
- **CreatedAt**: DateTime
- **UpdatedAt**: DateTime
- **Version**: String

## UserQuestion
- **QuestionText**: Text, required
- **AskedAt**: DateTime, required
- **UserId**: String (anonymous identifier)
- **ModuleContext**: Reference to CourseModules
- **ChapterContext**: Reference to Chapter
- **SourceContent**: Reference to BookContent

## ChatbotResponse
- **ResponseText**: Text, required
- **CreatedAt**: DateTime, required
- **QuestionId**: Reference to UserQuestion
- **Sources**: List of BookContent references
- **Confidence**: Float (0-1)
- **GroundedInBook**: Boolean

## Validation Rules
- CourseModules: Each module must have 3-4 chapters
- Chapter: ChapterNumber must be unique within a module
- Activity: Difficulty must match the target audience level
- CodeExample: Must be testable and runnable
- RAG Chatbot: Responses must reference BookContent entities only
- Humanoid Robot Simulation: Must be executable in Gazebo/Unity
- Sensor/Actuator: Must correspond to actual ROS interfaces
- UserQuestion: Cannot be empty
- ChatbotResponse: Cannot contain information not in referenced BookContent