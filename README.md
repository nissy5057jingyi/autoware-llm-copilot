# Autoware LLM Co-pilot

LLM-based natural language interface for Autoware Universe autonomous driving system.
Reproduces the framework from: "Modular Autonomy with Conversational Interaction" (Seegert et al., 2026)

## Architecture
- **Stage 1**: User instruction ? DSL command (via LLM)
- **Validation Node**: Whitelist-based safety check + ROS2 execution
- **Stage 2**: Execution result ? Natural language feedback (via LLM)

## Five Interaction Categories (Paper Table I)
| Category | Example |
|----------|---------|
| INFO | "What is the current speed?" |
| MISSION | "Start driving autonomously." |
| CONFIG | "Reduce your velocity to 30." |
| COOP | "Please go straight instead of turning left." |
| INTERVENTION | "The traffic light is green, you can go." |

## Setup

### Requirements
- Ubuntu 22.04
- ROS2 Humble
- Autoware Universe
- Python 3.10+

### Install dependencies
```bash
pip3 install pyyaml SpeechRecognition pydub
sudo apt install ffmpeg
```

### API Key
```bash
export OPENROUTER_API_KEY="your-key-here"
```
Get a free key at: https://openrouter.ai

## Usage

### Text input
```bash
source ~/autoware/install/setup.bash
python3 main.py "What is the current speed?"
```

### Voice input (m4a/wav file)
```bash
python3 main.py --file voice.m4a
```

### Run all five demo categories
```bash
python3 main.py --demo
```

## Files
- `main.py` - Two-stage LLM pipeline + voice input
- `validation_node.py` - DSL parser, validator, ROS2 executor
