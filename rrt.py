import numpy as np

from common import (
    ROOT,
    CollisionChecker,
    Q_MIN,
    Q_MAX,
    Q_START,
    Q_GOAL_REFERENCE,
    forward_kinematics,
    inverse_kinematics,
    shortcut_path,
    resample_path,
    save_excel,
)


def steer(start, target, step=0.32):
    direction = target - start
    distance = np.linalg.norm(direction)

    if distance <= step:
        return target.copy()

    return start + step * direction / distance


def rrt(start, goal, checker, iterations=8000):
    rng = np.random.default_rng(0)
    nodes = [start.copy()]
    parents = [-1]

    for _ in range(iterations):
        if rng.random() < 0.12:
            sample = goal
        else:
            sample = rng.uniform(Q_MIN, Q_MAX)
            sample = 0.8 * sample + 0.2 * goal

        nearest = int(np.argmin([
            np.linalg.norm(node - sample)
            for node in nodes
        ]))

        new_node = steer(
            nodes[nearest],
            sample,
        )

        if not checker.edge_is_valid(
            nodes[nearest],
            new_node,
        ):
            continue

        nodes.append(new_node)
        parents.append(nearest)

        if (
            np.linalg.norm(new_node - goal) <= 0.32
            and checker.edge_is_valid(new_node, goal)
        ):
            path = []
            index = len(nodes) - 1

            while index >= 0:
                path.append(nodes[index])
                index = parents[index]

            path = path[::-1]

            if np.linalg.norm(path[-1] - goal) > 1e-9:
                path.append(goal)

            return np.array(path)

    raise RuntimeError("RRT could not find a path")


if __name__ == "__main__":
    checker = CollisionChecker()

    target_pose = forward_kinematics(
        Q_GOAL_REFERENCE
    )

    q_goal = inverse_kinematics(
        target_pose,
        Q_START,
    )

    path = rrt(
        Q_START,
        q_goal,
        checker,
    )

    path = shortcut_path(path, checker)
    path = resample_path(path)

    output = ROOT / "results" / "rrt_path.xlsx"
    save_excel(path, output)

    print("Goal configuration:")
    print(q_goal)

    print("Samples:", len(path))
    print("Duration:", (len(path) - 1) * 0.01, "s")
    print("Output:", output)
