
import math
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from sac_environment import SACEnvironment


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "SAC_trained_agent.zip"
EVALUATION_FILE = ROOT / "results" / "evaluation.csv"

# Miền goal, số action và số bước tối đa của mỗi episode.
LEVELS = [
    (0.10,        100_000, 200),
    (0.20,         50_000, 220),
    (0.35,         50_000, 250),
    (0.55,         50_000, 300),
    (0.80,        150_000, 350),
    (1.10,        200_000, 400),
    (math.pi / 2, 400_000, 450),
]


def make_environment(level_index, evaluation=False):
    span, _, max_steps = LEVELS[level_index]
    environment = SACEnvironment(
        randomize_goal=True,
        goal_span=span,
        goal_spans=None if evaluation else [
            level[0] for level in LEVELS[:level_index + 1]
        ],
        stop_at_gate=True,
    )
    environment.MAX_STEPS = max_steps
    return environment


def evaluate(model):
    environment = make_environment(len(LEVELS) - 1, evaluation=True)
    rows = []

    for episode in range(20):
        observation, info = environment.reset(seed=700_000 + episode)
        reward_sum = 0.0
        attempted_collision = False

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = (
                environment.step(action)
            )
            reward_sum += reward
            attempted_collision |= info["collision"]
            if terminated or truncated:
                break

        rows.append({
            "point": episode + 1,
            "success": info["sac_gate_reached"],
            "attempted_collision": attempted_collision,
            "steps": environment.steps,
            "time_s": environment.steps * environment.SAMPLE_TIME,
            "reward": reward_sum,
            "position_error_m": info["position_error_m"],
            "orientation_error_deg": math.degrees(
                info["orientation_error_rad"]
            ),
            "clearance_m": info["clearance_m"],
            **{
                f"q_goal_{joint + 1}": info["goal_q"][joint]
                for joint in range(6)
            },
            "goal_x_m": info["goal_position"][0],
            "goal_y_m": info["goal_position"][1],
            "goal_z_m": info["goal_position"][2],
        })

    environment.close()
    table = pd.DataFrame(rows)
    EVALUATION_FILE.parent.mkdir(exist_ok=True)
    table.to_csv(EVALUATION_FILE, index=False)
    return table


def main():
    model = None
    environment = None

    for index, (span, actions, _) in enumerate(LEVELS):
        if environment is not None:
            environment.close()
        environment = Monitor(make_environment(index))

        if model is None:
            model = SAC(
                "MlpPolicy",
                environment,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=10_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                policy_kwargs={"net_arch": [256, 256]},
                seed=0,
                device="cpu",
                verbose=1,
            )
        else:
            model.set_env(environment)

        print(
            f"\nLevel {index + 1}/7: goal ±{span:.3f} rad, "
            f"train {actions:,} action"
        )
        model.learn(
            total_timesteps=actions,
            reset_num_timesteps=False,
            progress_bar=False,
        )

    environment.close()
    MODEL_FILE.parent.mkdir(exist_ok=True)
    model.save(MODEL_FILE)

    result = evaluate(model)
    print("\nTraining complete:", model.num_timesteps, "actions")
    print("Success:", f"{result['success'].mean():.0%}")
    print("Attempted collision:", f"{result['attempted_collision'].mean():.0%}")
    print("Model:", MODEL_FILE)
    print("Evaluation:", EVALUATION_FILE)


if __name__ == "__main__":
    main()
