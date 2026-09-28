import math
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

from rrt import steer


def is_ancestor(candidate, node, parents):
    while node >= 0:
        if node == candidate:
            return True

        node = parents[node]

    return False


def update_children(parent, nodes, parents, costs):
    for child in range(len(nodes)):
        if parents[child] == parent:
            costs[child] = (
                costs[parent]
                + np.linalg.norm(
                    nodes[child] - nodes[parent]
                )
            )

            update_children(
                child,
                nodes,
                parents,
                costs,
            )


def rrt_star(start, goal, checker, iterations=8000):
    rng = np.random.default_rng(0)

    nodes = [start.copy()]
    parents = [-1]
    costs = [0.0]
    goal_nodes = []

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

        count = len(nodes)

        radius = max(
            0.34,
            min(
                0.85,
                1.8
                * (
                    math.log(count + 1)
                    / (count + 1)
                ) ** (1 / 6),
            ),
        )

        neighbors = [
            i for i, node in enumerate(nodes)
            if np.linalg.norm(node - new_node) <= radius
        ]

        parent = nearest
        best_cost = (
            costs[nearest]
            + np.linalg.norm(
                new_node - nodes[nearest]
            )
        )

        for i in neighbors:
            new_cost = (
                costs[i]
                + np.linalg.norm(
                    new_node - nodes[i]
                )
            )

            if (
                new_cost < best_cost
                and checker.edge_is_valid(
                    nodes[i],
                    new_node,
                )
            ):
                parent = i
                best_cost = new_cost

        nodes.append(new_node)
        parents.append(parent)
        costs.append(best_cost)

        new_index = len(nodes) - 1

        for i in neighbors:
            if is_ancestor(
                i,
                new_index,
                parents,
            ):
                continue

            new_cost = (
                costs[new_index]
                + np.linalg.norm(
                    nodes[i] - new_node
                )
            )

            if (
                new_cost < costs[i]
                and checker.edge_is_valid(
                    new_node,
                    nodes[i],
                )
            ):
                parents[i] = new_index
                costs[i] = new_cost

                update_children(
                    i,
                    nodes,
                    parents,
                    costs,
                )

        if (
            np.linalg.norm(new_node - goal) <= radius
            and checker.edge_is_valid(new_node, goal)
        ):
            goal_nodes.append(new_index)

    if not goal_nodes:
        raise RuntimeError("RRT* could not find a path")

    best_goal = min(
        goal_nodes,
        key=lambda i: (
            costs[i]
            + np.linalg.norm(nodes[i] - goal)
        ),
    )

    path = [goal.copy()]
    index = best_goal

    while index >= 0:
        path.append(nodes[index])
        index = parents[index]

    return np.array(path[::-1])


def main():
    checker = CollisionChecker()

    target_pose = forward_kinematics(
        Q_GOAL_REFERENCE
    )

    q_goal = inverse_kinematics(
        target_pose,
        Q_START,
    )

    path = rrt_star(
        Q_START,
        q_goal,
        checker,
    )

    path = shortcut_path(path, checker)
    path = resample_path(path, max_step=0.01)

    output = ROOT / "results" / "rrt_star_path.xlsx"
    save_excel(path, output)

    print("Goal configuration:")
    print(q_goal)

    print("Samples:", len(path))
    print("Duration:", (len(path) - 1) * 0.01, "s")
    print("Output:", output)


if __name__ == "__main__":
    main()
