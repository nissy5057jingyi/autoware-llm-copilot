import json
import os
import urllib.request
import sys
from validation_node import process_dsl_command, get_av_status_text

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL   = "google/gemma-3-4b-it:free"

KNOWLEDGE_BASE = """You are an Autonomous Vehicle assistant using Autoware Universe.
Convert the user instruction into a DSL command in YAML format.
Output ONLY the YAML, no explanation, no markdown, no code blocks.

Categories and actions:
- INFO: GET_VELOCITY, GET_SPEED_LIMIT, GET_OPERATION_MODE, GET_ETA
- MISSION: START_DRIVE, STOP_DRIVE, SET_DESTINATION (params: destination)
- CONFIG: SET_PARAM (params: name=[max_vel|max_accel|headway], value=<number>)
- COOP: LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, TURN_LEFT, TURN_RIGHT, GO_STRAIGHT, OVERTAKE
- INTERVENTION: OVERRIDE_TRAFFIC_LIGHT (params: state=[green|red|yellow]), EMERGENCY_STOP, CLEAR_GHOST_OBJECT

Format:
command_type: <CATEGORY>
action: <ACTION>
parameters:
  - name: <NAME>
    value: <VALUE>

If no parameters needed:
parameters: []
"""

ICL_EXAMPLES = [
    {"role":"user","content":"What is the current speed?"},
    {"role":"assistant","content":"command_type: INFO\naction: GET_VELOCITY\nparameters: []"},
    {"role":"user","content":"Set the speed limit to 90."},
    {"role":"assistant","content":"command_type: CONFIG\naction: SET_PARAM\nparameters:\n  - name: max_vel\n    value: 90.0"},
    {"role":"user","content":"Start driving autonomously."},
    {"role":"assistant","content":"command_type: MISSION\naction: START_DRIVE\nparameters: []"},
    {"role":"user","content":"The traffic light is green, you can go."},
    {"role":"assistant","content":"command_type: INTERVENTION\naction: OVERRIDE_TRAFFIC_LIGHT\nparameters:\n  - name: state\n    value: green"},
    {"role":"user","content":"Please go straight instead of turning left."},
    {"role":"assistant","content":"command_type: COOP\naction: GO_STRAIGHT\nparameters: []"},
]

def call_llm(messages):
    payload = json.dumps({"model": MODEL, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()

def stage1_translate(instruction):
    system_prompt = KNOWLEDGE_BASE + "\nCurrent AV Status:\n" + get_av_status_text()
    messages = [{"role":"user","content":system_prompt}]
    messages += ICL_EXAMPLES
    messages.append({"role":"user","content":instruction})
    return call_llm(messages)

def stage2_feedback(instruction, result):
    if result["success"]:
        status = f"SUCCESS. Result: {result['result']}"
    else:
        status = f"FAILED. Reason: {result['reason']}"
    prompt = (
        f"The passenger said: \"{instruction}\"\n"
        f"Execution status: {status}\n\n"
        f"You must use the exact data from the result above in your reply.\n"
        f"Reply in one short friendly sentence. Do not use placeholders like [insert value here]."
    )
    return call_llm([{"role":"user","content":prompt}])

def run_pipeline(instruction, verbose=True):
    if verbose:
        print(f"\n{'='*55}")
        print(f"[Instruction] {instruction}")
        print(f"{'='*55}")
    if verbose:
        print("[Stage 1] Translating to DSL...")
    dsl = stage1_translate(instruction)
    if verbose:
        print(f"[DSL Output]\n{dsl}")
    if verbose:
        print("[Validation] Validating and executing...")
    result = process_dsl_command(dsl)
    if verbose:
        s = "OK" if result["success"] else "FAIL"
        print(f"[Execution] {s}: {result['result'] or result['reason']}")
    if verbose:
        print("[Stage 2] Generating feedback...")
    feedback = stage2_feedback(instruction, result)
    if verbose:
        print(f"[Passenger Feedback] {feedback}")
    return {"instruction":instruction,"dsl":dsl,"execution":result,"feedback":feedback}

DEMO = [
    "What is the current speed?",
    "Start driving autonomously.",
    "Reduce your velocity to 30.",
    "Please go straight instead of turning left.",
    "The traffic light is green, you can go.",
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--demo":
            print("Running all five interaction categories from the paper...\n")
            results = []
            for inst in DEMO:
                r = run_pipeline(inst)
                results.append(r)
            print(f"\n{'='*55}")
            print("Summary")
            print(f"{'='*55}")
            for r in results:
                s = "OK" if r["execution"]["success"] else "FAIL"
                print(f"[{s}] {r['instruction']}")
                print(f"     -> {r['feedback']}\n")
        elif mode == "--file" and len(sys.argv) > 2:
            import speech_recognition as sr
            import subprocess, tempfile, os
            input_file = sys.argv[2]
            print(f"[Voice] Reading file: {input_file}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
            subprocess.run(["ffmpeg", "-y", "-i", input_file, "-ar", "16000",
                           "-ac", "1", "-f", "wav", tmp_wav],
                          capture_output=True)
            r = sr.Recognizer()
            with sr.AudioFile(tmp_wav) as source:
                audio = r.record(source)
            os.unlink(tmp_wav)
            instruction = r.recognize_google(audio, language="en-US")
            print(f"[STT Result] {instruction}")
            run_pipeline(instruction)
        else:
            run_pipeline(" ".join(sys.argv[1:]))
    else:
        print("Usage:")
        print("  Text input:  python3 main.py \"your instruction\"")
        print("  All demos:   python3 main.py --demo")
        print("  Audio file:  python3 main.py --file audio.wav")
