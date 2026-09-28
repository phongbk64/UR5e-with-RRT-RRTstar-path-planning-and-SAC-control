
import math
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from common import (
    CollisionChecker,
    Q_GOAL_REFERENCE,
    Q_MAX,
    Q_MIN,
    Q_START,
    forward_kinematics,
    inverse_kinematics,
    resample_path,
)
from sac_environment import SACEnvironment


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "SAC_trained_agent.zip"
OUTPUT_FILE = ROOT / "results" / "sac1_ik_path.xlsx"
TARGET_Q = Q_GOAL_REFERENCE.copy()  # Thay goal tại đây khi cần.

def save_samples(rows, environment, phase):
    for q, position_error, angle_error, collision in (
        environment.last_command_samples
    ):
        rows.append({
            "phase": phase,
            **{f"q{joint + 1}": q[joint] for joint in range(6)},
            "position_error_m": position_error,
            "orientation_error_deg": math.degrees(angle_error),
            "collision": collision,
        })


def main():
    model = SAC.load(MODEL_FILE, device="cpu")
    checker = CollisionChecker()
    goal_pose = forward_kinematics(TARGET_Q)

    if not checker.state_is_valid(TARGET_Q):
        raise RuntimeError("The goal configuration is in collision.")

    environment = SACEnvironment(
        start_q=Q_START,
        goal_q=TARGET_Q,
        randomize_goal=False,
        stop_at_gate=False,
    )
    environment.MAX_STEPS = 450
    observation, _ = environment.reset(seed=0)
    rows = []

    for _ in range(environment.MAX_STEPS):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, _, _, info = environment.step(action)
        save_samples(rows, environment, "SAC")
        if info["sac_gate_reached"]:
            break
    else:
        environment.close()
        raise RuntimeError("SAC did not reach the handover region.")

    current_q = environment.data.qpos[:6].copy()
    ik_q = inverse_kinematics(goal_pose, current_q)
    if not checker.edge_is_valid(current_q, ik_q):
        environment.close()
        raise RuntimeError("The IK path from handover to goal is unsafe.")

    ik_path = resample_path(
        np.vstack([current_q, ik_q]),
        max_step=math.radians(2.0),
    )
    for q_reference in ik_path[1:]:
        observation, info = environment.apply_target(q_reference)
        save_samples(rows, environment, "IK")
        if info["collision"]:
            environment.close()
            raise RuntimeError("Collision during the IK phase.")

    command = ik_q.copy()
    distance, angle = environment.errors()
    for _ in range(environment.MAX_STEPS):
        if distance < 1e-5 and angle < 2e-4:
            break
        command = np.clip(
            command + 0.35 * (ik_q - environment.data.qpos[:6]),
            Q_MIN,
            Q_MAX,
        )
        observation, info = environment.apply_target(command)
        save_samples(rows, environment, "IK")
        if info["collision"]:
            environment.close()
            raise RuntimeError("Collision while settling at the goal.")
        distance, angle = environment.errors()

    environment.close()
    if distance >= 1e-5 or angle >= 2e-4:
        raise RuntimeError("IK did not reach the required tolerance.")

    table = pd.DataFrame(rows)
    table.insert(
        0,
        "time_s",
        np.arange(len(table)) * SACEnvironment.COMMAND_TIME,
    )
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    table.to_excel(OUTPUT_FILE, index=False)

    print(f"Position error: {distance:.6e} m")
    print(f"Orientation error: {math.degrees(angle):.6f} deg")
    print("Collision:", bool(table["collision"].any()))
    print("Trajectory:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
