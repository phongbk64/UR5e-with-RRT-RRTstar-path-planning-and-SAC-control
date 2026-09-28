## thông số và các phép tính dùng chung
# thông số UR5e
# Bảng D-H
# DH transform()
# forward kinematics()
# rotation_vector()
# Jacobian()
# inverse_kinematic()
# CollisionChecker

from pathlib import Path
import math

import mujoco
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "assets" / "scene.xml"

Q_MIN = np.full(6, -math.pi)
Q_MAX = np.full(6, math.pi)

Q_START = np.array([
    -1.5708,
    -1.5708,
     1.5708,
    -1.5708,
    -1.5708,
     0.0,
])

Q_GOAL_REFERENCE = np.array([
    -2.7,
    -0.7,
     0.7,
    -1.5,
    -1.4,
    -0.5,
])


# theta_offset, d, a, alpha
DH_TABLE = np.array([
    [-math.pi / 2, 0.163,  0.000,  math.pi / 2],
    [0.0,          0.000, -0.425,  0.0],
    [0.0,          0.000, -0.392,  0.0],
    [0.0,          0.134,  0.000,  math.pi / 2],
    [0.0,          0.100,  0.000, -math.pi / 2],
    [0.0,          0.100,  0.000,  0.0],
])


TOOL_TRANSFORM = np.array([
    [ 0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [ 0.0, 0.0, 1.0, 0.0],
    [ 0.0, 0.0, 0.0, 1.0],
])


def dh_transform(theta, d, a, alpha):
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,       sa,       ca,      d],
        [0.0,      0.0,      0.0,    1.0],
    ])


def forward_kinematics(q):
    transform = np.eye(4)

    for qi, (offset, d, a, alpha) in zip(q, DH_TABLE):
        transform = transform @ dh_transform(
            qi + offset,
            d,
            a,
            alpha,
        )

    return transform @ TOOL_TRANSFORM


def rotation_vector(current, target):
    relative = target @ current.T

    angle = math.acos(
        float(
            np.clip(
                (np.trace(relative) - 1) / 2,
                -1,
                1,
            )
        )
    )

    if angle < 1e-10:
        return np.zeros(3)

    if math.pi - angle < 1e-5:
        values, vectors = np.linalg.eig(relative)
        axis = np.real(
            vectors[:, np.argmin(np.abs(values - 1))]
        )

        return angle * axis / np.linalg.norm(axis)

    axis = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) / (2 * math.sin(angle))

    return angle * axis


def jacobian(q):
    transform = np.eye(4)
    origins = []
    axes = []

    for qi, (offset, d, a, alpha) in zip(q, DH_TABLE):
        origins.append(
            transform[:3, 3].copy()
        )

        axes.append(
            transform[:3, 2].copy()
        )

        transform = transform @ dh_transform(
            qi + offset,
            d,
            a,
            alpha,
        )

    end_position = forward_kinematics(q)[:3, 3]
    result = np.zeros((6, 6))

    for i in range(6):
        result[:3, i] = np.cross(
            axes[i],
            end_position - origins[i],
        )

        result[3:, i] = axes[i]

    return result


def inverse_kinematics(target, initial_q):
    q = np.array(initial_q, dtype=float)

    for _ in range(500):
        current = forward_kinematics(q)

        error = np.r_[
            target[:3, 3] - current[:3, 3],
            rotation_vector(
                current[:3, :3],
                target[:3, :3],
            ),
        ]

        position_error = np.linalg.norm(
            error[:3]
        )

        orientation_error = np.linalg.norm(
            error[3:]
        )

        if (
            position_error < 1e-5
            and orientation_error < 2e-4
        ):
            return q

        J = jacobian(q)

        step = J.T @ np.linalg.solve(
            J @ J.T + 0.03**2 * np.eye(6),
            error,
        )

        max_step = np.max(np.abs(step))

        if max_step > 0.2:
            step *= 0.2 / max_step

        q = np.clip(
            q + step,
            Q_MIN,
            Q_MAX,
        )

    raise RuntimeError("IK did not converge")


class CollisionChecker:
    def __init__(self, margin=0.04):
        self.model = mujoco.MjModel.from_xml_path(
            str(SCENE_XML)
        )

        self.data = mujoco.MjData(
            self.model
        )

        self.obstacles = {
            i for i in range(self.model.ngeom)
            if (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    i,
                ) or ""
            ).startswith("obstacle")
        }

        for i in self.obstacles:
            self.model.geom_size[i] += margin

    def state_is_valid(self, q):
        if (
            np.any(q < Q_MIN)
            or np.any(q > Q_MAX)
        ):
            return False

        self.data.qpos[:6] = q
        mujoco.mj_forward(
            self.model,
            self.data,
        )

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if (
                contact.geom1 in self.obstacles
                or contact.geom2 in self.obstacles
            ):
                return False

        return True

    def edge_is_valid(
        self,
        start,
        end,
        resolution=0.04,
    ):
        distance = np.max(
            np.abs(end - start)
        )

        count = max(
            1,
            math.ceil(distance / resolution),
        )

        for ratio in np.linspace(
            0,
            1,
            count + 1,
        ):
            q = start + ratio * (end - start)

            if not self.state_is_valid(q):
                return False

        return True


def shortcut_path(path, checker):
    rng = np.random.default_rng(1)
    path = list(path)

    for _ in range(250):
        if len(path) <= 2:
            break

        i, j = sorted(
            rng.choice(
                len(path),
                2,
                replace=False,
            )
        )

        if (
            j > i + 1
            and checker.edge_is_valid(
                path[i],
                path[j],
            )
        ):
            path = path[:i + 1] + path[j:]

    return np.array(path)


def resample_path(path, max_step=0.01):
    result = [path[0]]

    for start, end in zip(
        path[:-1],
        path[1:],
    ):
        distance = np.max(
            np.abs(end - start)
        )

        count = max(
            1,
            math.ceil(distance / max_step),
        )

        for ratio in np.linspace(
            0,
            1,
            count + 1,
        )[1:]:
            result.append(
                start + ratio * (end - start)
            )

    return np.array(result)


def save_excel(path, filename):
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.DataFrame(
        path,
        columns=[
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
        ],
    )

    data.insert(
        0,
        "time_s",
        np.arange(len(path)) * 0.01,
    )

    data.to_excel(
        filename,
        index=False,
    )
