# Multi-Agent Systems with Google ADK

This project was completed as a Google ADK lab.

## Systems built

- Travel planner with specialist agent handoffs and saved attractions.
- Movie pitch workflow using sequential, loop, and parallel agents.
- Researcher, screenwriter, critic, casting, box office, and file writer roles.

## Setup

Use Python 3.12 and a Google Cloud project with billing and the
aiplatform.googleapis.com API enabled.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env
```

Edit .env with your project settings, then copy it into both agent folders:

```bash
cp .env parent_and_subagents/.env
cp .env workflow_agents/.env
set -a
source .env
set +a
```

## Run in the terminal

```bash
adk run parent_and_subagents
adk run workflow_agents
```

Run one agent at a time.

## Run in Cloud Shell Web Preview

```bash
adk web --allow_origins 'regex:https://.*\.cloudshell\.dev' --reload_agents --port 8010
```

Open Web Preview on port 8010 and select an agent.

## Output

Generated movie pitch files are saved in movie_pitches.

## Learning

The lab covers agent delegation, session state, iterative review,
parallel tasks, and tool permissions.

## Reference

Adapted from Google's Build Multi-Agent Systems with ADK Codelab:
https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk

Original source notices and licenses are retained.