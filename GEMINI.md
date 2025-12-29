# GEMINI.md - Project Implementation Guidelines

This document provides technical details about the project structure and rules for development that all AI agents and developers should follow.

## Project Vision

FunctionGemma Toolkit is designed to bridge the gap between LLM fine-tuning and web-based execution. It emphasizes ease of use for game developers while providing powerful training capabilities.

## Directory Structure

- `trainer/`: Contains the Flask/FastAPI backend and React/Vue frontend for the training web tool. Uses Docker for environment isolation.
- `packages/lib-web-game/`: A TypeScript/JavaScript library that wraps `transformers.js`. It handles model loading, prompt templating for FunctionGemma, and ONNX runtime optimization.
- `examples/`: Contains end-to-end examples, such as an "Inventory Management" demo or "NPC Dialog" demo.
- `docs/`: Technical documentation, architectural diagrams, and research notes.

## Development Rules

### Language Requirement
- **All documentation, comments, and commit messages MUST be in English.**

### Task and Issue Management
- We use **GitHub CLI (`gh`)** for all task management.
- **Tasks**: Every significant task must be tracked as a GitHub Issue.
- **Implementation**: All code changes must be submitted via a Pull Request (PR) linked to an issue.
- **Workflow**:
    1. Check for existing issues using `gh issue list`.
    2. Create a new issue if none exists: `gh issue create --title "..." --body "..."`.
    3. Work on the implementation.
    4. Create a PR: `gh pr create --title "..." --body "..."`.

### Technical Constraints
- The library must be distributable as an NPM package.
- The trainer must remain self-contained within Docker.
- Models should default to ONNX format for the web library.

## AI Agent Instructions

When working in this repository:
- Always check `GEMINI.md` (this file) and `task.md` (in the workspace) before starting a new task.
- Follow the GitHub CLI process for issues and PRs.
- Maintain the clean directory structure described above.
