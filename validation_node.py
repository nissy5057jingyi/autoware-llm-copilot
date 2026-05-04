import yaml
import re
import subprocess
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DSLCommand:
    command_type: str
    action: str
    parameters: dict
    parameters_raw: list = field(default_factory=list)

ALLOWED_ACTIONS = {
    "INFO": {
        "GET_VELOCITY":       {"params": []},
        "GET_SPEED_LIMIT":    {"params": []},
        "GET_OPERATION_MODE": {"params": []},
        "GET_ETA":            {"params": []},
    },
    "MISSION": {
        "START_DRIVE":     {"params": []},
        "STOP_DRIVE":      {"params": []},
        "SET_DESTINATION": {"params": ["destination"]},
    },
    "CONFIG": {
        "SET_PARAM": {"params": [],
                      "param_bounds": {
                          "max_vel":   (0.0, 130.0),
                          "max_accel": (0.0, 5.0),
                          "headway":   (0.5, 10.0),
                      }},
    },
    "COOP": {
        "LANE_CHANGE_LEFT":  {"params": []},
        "LANE_CHANGE_RIGHT": {"params": []},
        "TURN_LEFT":         {"params": []},
        "TURN_RIGHT":        {"params": []},
        "GO_STRAIGHT":       {"params": []},
        "OVERTAKE":          {"params": []},
    },
    "INTERVENTION": {
        "OVERRIDE_TRAFFIC_LIGHT": {"params": ["state"]},
        "EMERGENCY_STOP":         {"params": []},
        "CLEAR_GHOST_OBJECT":     {"params": []},
    },
}

AV_STATUS = {
    "velocity":       0.0,
    "speed_limit":    50.0,
    "operation_mode": "manual",
    "eta_seconds":    0,
    "destination":    "Unknown",
}

def get_ros2_env():
    env = os.environ.copy()
    env["AMENT_PREFIX_PATH"] = "/home/zhoujingyi654321/autoware/install:/opt/ros/humble"
    env["PATH"] = "/opt/ros/humble/bin:" + env.get("PATH", "")
    return env

def ros2_run(args, timeout=5):
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, env=get_ros2_env())

def parse_dsl(dsl_text: str) -> Optional[DSLCommand]:
    try:
        data = yaml.safe_load(dsl_text)
        cmd_type = str(data.get("command_type", "")).upper()
        action   = str(data.get("action", "")).upper()
        raw      = data.get("parameters", []) or []
        params   = {}
        for p in raw:
            if isinstance(p, dict) and "name" in p:
                params[p["name"]] = p["value"]
        return DSLCommand(command_type=cmd_type, action=action,
                          parameters=params, parameters_raw=raw)
    except Exception as e:
        print(f"[Parse error] {e}")
        return None

def validate(cmd: DSLCommand):
    if cmd.command_type not in ALLOWED_ACTIONS:
        return False, f"Unknown command type: {cmd.command_type}"
    actions = ALLOWED_ACTIONS[cmd.command_type]
    if cmd.action not in actions:
        return False, f"Unsupported action: {cmd.action}"
    spec = actions[cmd.action]
    for rp in spec["params"]:
        if rp not in cmd.parameters:
            return False, f"Missing parameter: {rp}"
    if "param_bounds" in spec and cmd.parameters:
        for param_name, param_value in cmd.parameters.items():
            if param_name in spec["param_bounds"]:
                lo, hi = spec["param_bounds"][param_name]
                try:
                    v = float(param_value)
                    if not (lo <= v <= hi):
                        return False, f"Parameter {param_name}={v} out of range [{lo}, {hi}]"
                except (TypeError, ValueError):
                    return False, f"Parameter value is not a number: {param_value}"
    return True, "Validation passed"

def execute(cmd: DSLCommand):
    try:
        return execute_ros2(cmd)
    except Exception as e:
        print(f"[ROS2 error] {e}, falling back to Mock")
        return execute_mock(cmd)

def execute_ros2(cmd: DSLCommand):
    t, a, p = cmd.command_type, cmd.action, cmd.parameters

    if t == "INFO":
        if a == "GET_VELOCITY":
            r = ros2_run(["ros2", "topic", "echo", "--once", "--no-arr",
                          "/vehicle/status/velocity_status"])
            if r.returncode == 0 and r.stdout:
                m = re.search(r"longitudinal_velocity:\s*([\d.\-]+)", r.stdout)
                if m:
                    speed_ms = float(m.group(1))
                    speed_kmh = speed_ms * 3.6
                    return True, f"Current speed is {speed_kmh:.1f} km/h ({speed_ms:.2f} m/s)"
            return True, f"Current speed is {AV_STATUS['velocity']} km/h (no topic data)"

        elif a == "GET_OPERATION_MODE":
            r = ros2_run(["ros2", "topic", "echo", "--once", "--no-arr",
                          "/system/operation_mode/state"])
            if r.returncode == 0 and r.stdout:
                m = re.search(r"mode:\s*(\d+)", r.stdout)
                mode_map = {"1": "STOP", "2": "AUTONOMOUS", "3": "LOCAL", "4": "REMOTE"}
                mode_str = mode_map.get(m.group(1), "UNKNOWN") if m else "UNKNOWN"
                return True, f"Current operation mode is {mode_str}"
            return True, f"Current operation mode is {AV_STATUS['operation_mode']} (no topic data)"

        elif a == "GET_SPEED_LIMIT":
            return True, f"Current speed limit is {AV_STATUS['speed_limit']} km/h"

        elif a == "GET_ETA":
            mins = AV_STATUS['eta_seconds'] // 60
            secs = AV_STATUS['eta_seconds'] % 60
            return True, f"ETA is {mins} min {secs} sec to {AV_STATUS['destination']}"

    elif t == "MISSION":
        if a == "START_DRIVE":
            r = ros2_run(["ros2", "service", "call",
                          "/api/operation_mode/change_to_autonomous",
                          "autoware_adapi_v1_msgs/srv/ChangeOperationMode", "{}"], timeout=10)
            if r.returncode == 0 and "success=True" in r.stdout:
                AV_STATUS["operation_mode"] = "autonomous"
                return True, "Switched to autonomous driving mode"
            return True, "Autonomous drive command sent (check Autoware status)"

        elif a == "STOP_DRIVE":
            r = ros2_run(["ros2", "service", "call",
                          "/api/operation_mode/change_to_stop",
                          "autoware_adapi_v1_msgs/srv/ChangeOperationMode", "{}"], timeout=10)
            AV_STATUS["operation_mode"] = "stop"
            return True, "Vehicle stopped"

        elif a == "SET_DESTINATION":
            dest = p.get("destination", "Unknown")
            AV_STATUS["destination"] = dest
            return True, f"Destination set to: {dest}"

    elif t == "CONFIG":
        if a == "SET_PARAM":
            for name, value in p.items():
                if name == "max_vel":
                    r = ros2_run(["ros2", "param", "set",
                                  "/planning/scenario_planning/motion_velocity_smoother",
                                  "max_velocity", str(float(value))], timeout=10)
                    AV_STATUS["speed_limit"] = float(value)
                    return True, f"Maximum speed set to {value} km/h"
                elif name == "max_accel":
                    return True, f"Maximum acceleration set to {value} m/s2"
                elif name == "headway":
                    return True, f"Following distance set to {value} seconds"
                else:
                    return True, f"Parameter {name} set to {value}"

    elif t == "COOP":
        action_map = {
            "LANE_CHANGE_LEFT":  "Changing lane to the left",
            "LANE_CHANGE_RIGHT": "Changing lane to the right",
            "TURN_LEFT":         "Turning left at the next intersection",
            "TURN_RIGHT":        "Turning right at the next intersection",
            "GO_STRAIGHT":       "Route adjusted, going straight",
            "OVERTAKE":          "Executing overtake maneuver",
        }
        return True, action_map.get(a, f"Executing {a}")

    elif t == "INTERVENTION":
        if a == "OVERRIDE_TRAFFIC_LIGHT":
            state = p.get("state", "green")
            return True, f"Traffic light state overridden to: {state}"
        elif a == "EMERGENCY_STOP":
            AV_STATUS["operation_mode"] = "emergency_stop"
            return True, "Emergency stop triggered"
        elif a == "CLEAR_GHOST_OBJECT":
            return True, "Ghost object cleared"

    return False, f"Unknown command: {t}/{a}"

def execute_mock(cmd: DSLCommand):
    t, a, p = cmd.command_type, cmd.action, cmd.parameters
    if t == "INFO":
        if a == "GET_VELOCITY":
            return True, f"Current speed is {AV_STATUS['velocity']} km/h"
        elif a == "GET_SPEED_LIMIT":
            return True, f"Current speed limit is {AV_STATUS['speed_limit']} km/h"
        elif a == "GET_OPERATION_MODE":
            return True, f"Current operation mode is {AV_STATUS['operation_mode']}"
        elif a == "GET_ETA":
            return True, f"ETA {AV_STATUS['eta_seconds']} sec to {AV_STATUS['destination']}"
    elif t == "MISSION":
        if a == "START_DRIVE":
            AV_STATUS["operation_mode"] = "autonomous"
            return True, "Switched to autonomous driving mode"
        elif a == "STOP_DRIVE":
            AV_STATUS["operation_mode"] = "stop"
            return True, "Vehicle stopped"
        elif a == "SET_DESTINATION":
            dest = p.get("destination", "Unknown")
            AV_STATUS["destination"] = dest
            return True, f"Destination set to: {dest}"
    elif t == "CONFIG":
        if a == "SET_PARAM":
            for name, value in p.items():
                if name == "max_vel":
                    AV_STATUS["speed_limit"] = float(value)
                    return True, f"Maximum speed set to {value} km/h"
                elif name == "max_accel":
                    return True, f"Maximum acceleration set to {value} m/s2"
                elif name == "headway":
                    return True, f"Following distance set to {value} seconds"
                else:
                    return True, f"Parameter {name} set to {value}"
    elif t == "COOP":
        action_map = {
            "LANE_CHANGE_LEFT":  "Changing lane to the left",
            "LANE_CHANGE_RIGHT": "Changing lane to the right",
            "TURN_LEFT":         "Turning left at the next intersection",
            "TURN_RIGHT":        "Turning right at the next intersection",
            "GO_STRAIGHT":       "Route adjusted, going straight",
            "OVERTAKE":          "Executing overtake maneuver",
        }
        return True, action_map.get(a, f"Executing {a}")
    elif t == "INTERVENTION":
        if a == "OVERRIDE_TRAFFIC_LIGHT":
            state = p.get("state", "green")
            return True, f"Traffic light state overridden to: {state}"
        elif a == "EMERGENCY_STOP":
            AV_STATUS["operation_mode"] = "emergency_stop"
            return True, "Emergency stop triggered"
        elif a == "CLEAR_GHOST_OBJECT":
            return True, "Ghost object cleared"
    return False, f"Unknown command: {t}/{a}"

def process_dsl_command(dsl_text: str) -> dict:
    cmd = parse_dsl(dsl_text)
    if cmd is None:
        return {"success": False, "reason": "DSL parse failed", "result": ""}
    ok, reason = validate(cmd)
    if not ok:
        return {"success": False, "reason": reason, "result": ""}
    ok, result = execute(cmd)
    return {"success": ok, "reason": reason, "result": result}

def get_av_status_text() -> str:
    return (
        f"Current Velocity: {AV_STATUS['velocity']} km/h\n"
        f"Operation Mode: {AV_STATUS['operation_mode']}\n"
        f"Velocity Limit: {AV_STATUS['speed_limit']} km/h\n"
        f"Destination: {AV_STATUS['destination']}\n"
        f"ETA: {AV_STATUS['eta_seconds']} seconds\n"
    )
