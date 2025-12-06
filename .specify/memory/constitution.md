<!-- SYNC IMPACT REPORT
Version change: N/A → 1.0.0
Added sections: All principles and sections as defined below
Removed sections: None
Templates requiring updates: ✅ No updates needed - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md, .qwen/commands/*.toml
Follow-up TODOs: None
Modified principles: None
-->
# Physical AI & Humanoid Robotics Course Constitution

## Core Principles

### I. Accuracy
All explanations must reflect correct robotics, AI, and simulation engineering concepts. Content must be technically sound, factually accurate, and represent real-world implementations of ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.

### II. Clarity
Use beginner-to-intermediate–friendly explanations, visuals, and examples. All content must be accessible to students new to robotics and AI, with clear language, well-defined terminology, and helpful analogies that illuminate complex concepts.

### III. Depth & Rigor
Go deep into ROS 2, simulation, Isaac, VLA pipelines, and humanoid mechanics. Each topic must cover theoretical foundations, practical implementations, and real-world applications with sufficient technical detail for students to build actual systems.

### IV. Traceability
Each section must have a clear purpose and refer back to the core curriculum. All content must connect explicitly to the four core modules (ROS 2, Simulation, AI-Brain, VLA) and the final capstone Autonomous Humanoid project.

### V. Spec-Driven Workflow
Every step of the project (chapters, diagrams, API code, chatbot workflows) must be generated via: /sp.specify → Requirements, /sp.plan → Strategy, /sp.tasks → Task breakdown, /sp.implementation → Final structured output. This ensures consistency, traceability, and quality across all deliverables.

### VI. Zero Ambiguity
All specs must be deterministic, repeatable, and machine-actionable. Documentation and code must be precise enough that any developer can reproduce the exact same results, with no room for interpretation or guesswork in implementation steps.

## Constraints and Standards

### Technology Stack Compliance
- Follow Docusaurus folder structures and documentation standards for book creation.
- Use simple, understandable English appropriate for international audiences.
- No hallucinated technologies or APIs - all implementations must be grounded in existing, accessible tools.
- All RAG outputs must be grounded in book data with citations and references.
- Code must be runnable and deployable as-is with minimal setup requirements.
- Deployed book must function on GitHub Pages with embedded chatbot capabilities.

### Content Quality Standards
- All explanations must be technically accurate and validated against real-world implementations.
- Hands-on labs must include step-by-step workflows with expected outcomes.
- Visual aids (diagrams, charts, code examples) must clarify complex concepts.
- Content must be structured by learning progression, building from fundamentals to advanced topics.

## Development Workflow

### Book Creation Process
- Written using Docusaurus documentation framework.
- Structured by AI through Spec-Kit Plus for consistency and quality.
- Authored using Claude Code for technical accuracy and pedagogical effectiveness.
- Deployed to GitHub Pages with responsive design for accessibility.
- Includes hands-on labs covering ROS 2, Gazebo, and Isaac Sim practical implementations.

### RAG Chatbot System
- Embedded inside the published book interface for seamless user experience.
- Built with OpenAI Agents / ChatKit SDK for robust question answering.
- Implemented with FastAPI backend for scalable, reliable service.
- Utilizes Neon Serverless Postgres for logs and conversation memory.
- Powered by Qdrant Cloud Free Tier for embeddings and vector search.
- Answers must derive exclusively from book content to prevent hallucinations.
- Capabilities include full-book comprehension and focused text-selection responses.

### Curriculum Structure
- Organized around four core modules: ROS 2, Simulation, AI-Brain, VLA.
- Culminates in Autonomous Humanoid capstone project integrating all previous concepts.
- Each module builds systematically toward student ability to create simulated humanoid robots.
- Robot abilities include receiving voice commands, using AI brains with ROS 2, navigating and manipulating objects in simulation.

## Governance

This constitution governs all project activities and supersedes any conflicting practices. All contributions must align with these principles to maintain educational quality and technical coherence. Amendments require documentation of rationale, approval by project maintainers, and migration plans for affected content. All specifications, plans, tasks, and implementations must explicitly verify compliance with these principles. All pull requests and reviews must validate constitutional adherence before merging.

**Version**: 1.0.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-07