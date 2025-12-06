import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the Physical AI & Humanoid Robotics Course
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 - The Robotic Nervous System',
      items: [
        'modules/ros2/intro',
        'modules/ros2/chapter1',
        'modules/ros2/chapter2',
        'modules/ros2/chapter3',
        'modules/ros2/chapter4',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity - The Digital Twin',
      items: [
        'modules/simulation/intro',
        'modules/simulation/chapter1',
        'modules/simulation/chapter2',
        'modules/simulation/chapter3',
        'modules/simulation/chapter4',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac - The AI-Robot Brain',
      items: [
        'modules/isaac/intro',
        'modules/isaac/chapter1',
        'modules/isaac/chapter2',
        'modules/isaac/chapter3',
        'modules/isaac/chapter4',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'modules/vla/intro',
        'modules/vla/chapter1',
        'modules/vla/chapter2',
        'modules/vla/chapter3',
        'modules/vla/capstone',
      ],
    },
  ],
};

export default sidebars;
