# Story Backend

## Overview
This project is a backend service for generating 3D scenes and videos from text descriptions. It leverages two main tools:

1. **Three.js Tool**: Generates 3D scenes using Three.js
2. **Cosmos Tool**: Generates 3D scenes using NVIDIA's Cosmos API

## Architecture
The project uses a graph-based workflow system with nodes for different processing steps. It leverages Language Models (LLMs) like NVIDIA and Claude for enhancing prompts before generating content.

## Frameworks and Technologies

### Core Frameworks
- **LangGraph**: Powers the workflow orchestration with directed graphs and state management
  - Uses `StateGraph` for defining workflow nodes and edges
  - Implements `Pregel` for distributed graph processing

### Language Models
- **LangChain**: Provides the foundation for working with LLMs
  - Uses `BaseChatModel` for standardized LLM interactions
  - Implements message handling with `SystemMessage` and `HumanMessage`

### Data Validation
- **Pydantic**: Handles data validation and settings management
  - Implements `BaseModel` for type-safe data structures
  - Provides schema validation for API inputs and outputs

### Asynchronous Processing
- **AsyncResult**: Custom implementation for handling asynchronous tasks
  - Manages polling and result retrieval from external APIs
  - Implements timeout and error handling mechanisms

### External APIs
- **NVIDIA Cosmos API**: Generates 3D scenes from text descriptions
- **Three.js**: Creates interactive 3D visualizations in the browser

## Features
- **Prompt Upsampling**: Enhances simple text descriptions into detailed scene descriptions
- **Asynchronous Processing**: Handles video generation requests asynchronously
- **Error Handling**: Gracefully handles failures with fallback options

## Environment Variables
- `NVIDIA_API_KEY`: Required for authenticating requests to the NVIDIA Cosmos API

## Note
The documentation and code for this project were generated and enhanced using Windsurf, an agentic IDE powered by Codeium's AI Flow paradigm.

## Getting Started
To use this backend service, ensure you have the required API keys set in your environment and follow the API documentation for making requests to the service.