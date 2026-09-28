"""Môi trường SAC goal-conditioned cho UR5e với vật cản cố định."""

import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from common import (
    Q_MAX,
    Q_MIN,
    Q_START,
    SCENE_XML,
    forward_kinematics,
    rotation_vector,
)


class SACEnvironment(gym.Env):
    # Chu kỳ mô phỏng và điều khiển
    SAMPLE_TIME = 0.05       # Actor SAC: 20 Hz
    COMMAND_TIME = 0.01      # Lệnh actuator: 100 Hz
    PHYSICS_TIME = 0.002     # MuJoCo: 500 Hz

    # Giới hạn hành động
    MAX_DELTA_Q = 0.025      # rad / action
    MAX_STEPS = 300

    # Vùng chuyển giao SAC -> IK. Góc vẫn được học trong reward nhưng
    # không khóa chuyển trạng thái; IK sẽ hiệu chỉnh hướng cuối cùng.
    POSITION_GATE = 0.10     # m
    ANGLE_GATE = math.radians(10.0)

    # Khoảng cách an toàn với vật cản
    SAFETY_DISTANCE = 0.08   # m

    # Điều kiện lấy mẫu goal khi huấn luyện
    MIN_GOAL_DISTANCE = 0.105  # m, goal phải nằm ngoài vùng chuyển giao
    MIN_GOAL_HEIGHT = 0.08    # m, tránh goal quá thấp

    def __init__(
        self,
        start_q=None,
        goal_q=None,
        randomize_goal=True,
        goal_span=0.50,
        goal_spans=None,
        stop_at_gate=True,
    ):
        super().__init__()

        self.start_q = np.asarray(
            Q_START if start_q is None else start_q,
            dtype=float,
        ).copy()

        self.fixed_goal_q = (
            None if goal_q is None
            else np.asarray(goal_q, dtype=float).copy()
        )
        self.randomize_goal = bool(randomize_goal)
        self.goal_span = float(goal_span)
        self.goal_spans = (
            None if goal_spans is None
            else [float(value) for value in goal_spans]
        )
        self.stop_at_gate = bool(stop_at_gate)

        # ------------------------------------------------------------------
        # MuJoCo model
        # ------------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.check_data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.PHYSICS_TIME

        # Giữ cách cài đặt actuator giống chương trình trước.
        self.model.actuator_gainprm[:, 0] *= 4.0
        self.model.actuator_biasprm[:, 1] *= 4.0
        self.model.actuator_biasprm[:, 2] *= 2.0
        self.model.actuator_forcerange[:] *= 4.0

        self.command_steps = round(
            self.SAMPLE_TIME / self.COMMAND_TIME
        )
        self.physics_steps = round(
            self.COMMAND_TIME / self.PHYSICS_TIME
        )

        self.site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "attachment_site",
        )

        # Vật cản được đọc trực tiếp từ scene.xml và luôn giữ cố định.
        self.obstacle_ids = sorted(
            [
                geom_id
                for geom_id in range(self.model.ngeom)
                if self._geom_name(geom_id).startswith("obstacle")
            ],
            key=self._geom_name,
        )

        self.robot_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                self.model.geom_contype[geom_id]
                and geom_id not in self.obstacle_ids
                and self._geom_name(geom_id) != "floor"
            )
        ]

        # Observation được chuẩn hóa về cùng cỡ giá trị:
        # sin(q), cos(q), q_dot: 18
        # position error, rotation error: 6
        # vị trí tương đối của N vật cản: 3*N
        # khoảng hở nhỏ nhất: 1
        # Với hai vật cản trong scene.xml: 31 chiều.
        self.observation_dim = 25 + 3 * len(self.obstacle_ids)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        if self.fixed_goal_q is not None:
            self.goal_q = self.fixed_goal_q.copy()
            self.goal = forward_kinematics(self.goal_q)
        else:
            # Giá trị tạm trước lần reset đầu tiên.
            self.goal_q = self.start_q.copy()
            self.goal = forward_kinematics(self.goal_q)

        self.steps = 0
        self.last_command_samples = []

    # ----------------------------------------------------------------------
    # Các hàm cơ bản
    # ----------------------------------------------------------------------
    def _geom_name(self, geom_id):
        name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )
        return name or ""

    def _pose(self):
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[
            self.site_id
        ].reshape(3, 3).copy()
        return position, rotation

    def errors(self):
        position, rotation = self._pose()

        distance = np.linalg.norm(
            self.goal[:3, 3] - position
        )
        angle = np.linalg.norm(
            rotation_vector(
                rotation,
                self.goal[:3, :3],
            )
        )

        return float(distance), float(angle)

    # ----------------------------------------------------------------------
    # Collision / clearance
    # ----------------------------------------------------------------------
    def _clearance_for_data(self, data):
        if not self.obstacle_ids:
            return 10.0

        distances = [
            mujoco.mj_geomDistance(
                self.model,
                data,
                robot_id,
                obstacle_id,
                2.0,
                None,
            )
            for robot_id in self.robot_geom_ids
            for obstacle_id in self.obstacle_ids
        ]

        return float(min(distances))

    def _clearance(self):
        return self._clearance_for_data(self.data)

    def _collision_for_data(self, data):
        for index in range(data.ncon):
            contact = data.contact[index]

            if (
                contact.geom1 in self.obstacle_ids
                or contact.geom2 in self.obstacle_ids
            ):
                return True

        return False

    def _collision(self):
        return self._collision_for_data(self.data)

    def _configuration_is_safe(self, q):
        q = np.asarray(q, dtype=float)

        if np.any(q < Q_MIN) or np.any(q > Q_MAX):
            return False

        mujoco.mj_resetData(self.model, self.check_data)
        self.check_data.qpos[:6] = q
        self.check_data.qvel[:6] = 0.0
        self.check_data.ctrl[:6] = q
        mujoco.mj_forward(self.model, self.check_data)

        return not self._collision_for_data(self.check_data)

    # ----------------------------------------------------------------------
    # Random goal
    # ----------------------------------------------------------------------
    def _sample_goal_configuration(self):
        """Sinh một goal mới từ joint space nhưng bảo đảm goal khả thi.

        Khi goal_span < pi, mỗi khớp được random quanh start_q trong
        khoảng +/- goal_span. Khi goal_span đạt pi, toàn bộ miền Q_MIN-Q_MAX
        được sử dụng.
        """
        start_pose = forward_kinematics(self.start_q)

        sample_span = self.goal_span
        if self.goal_spans:
            # Ở miền lớn, tập trung 80% episode vào level hiện tại.
            # 20% còn lại ôn các level cũ để hạn chế forgetting.
            previous_spans = [
                value for value in self.goal_spans
                if 0.20 <= value < self.goal_span - 1e-9
            ]
            if not previous_spans or self.np_random.random() < 0.80:
                sample_span = self.goal_span
            else:
                sample_span = float(
                    self.np_random.choice(previous_spans)
                )

        for _ in range(1000):
            if sample_span >= math.pi - 1e-6:
                q_goal = self.np_random.uniform(Q_MIN, Q_MAX)
            else:
                lower = np.maximum(
                    Q_MIN,
                    self.start_q - sample_span,
                )
                upper = np.minimum(
                    Q_MAX,
                    self.start_q + sample_span,
                )
                q_goal = self.np_random.uniform(lower, upper)

            goal_pose = forward_kinematics(q_goal)
            position_distance = np.linalg.norm(
                goal_pose[:3, 3] - start_pose[:3, 3]
            )

            # Loại các goal quá dễ hoặc không phù hợp với môi trường.
            if position_distance < self.MIN_GOAL_DISTANCE:
                continue
            if goal_pose[2, 3] < self.MIN_GOAL_HEIGHT:
                continue
            if not self._configuration_is_safe(q_goal):
                continue

            return q_goal

        raise RuntimeError(
            "Could not sample a valid goal. "
            "Reduce goal_span or check scene.xml."
        )

    # ----------------------------------------------------------------------
    # Observation
    # ----------------------------------------------------------------------
    def _obstacle_features(self, end_position):
        """Vị trí tâm mỗi vật cản tương đối so với end-effector.

        Vật cản cố định trong scene.xml, nhưng vector (p_obstacle - p_ee)
        vẫn thay đổi theo chuyển động của robot.
        """
        features = []

        for geom_id in self.obstacle_ids:
            center = self.data.geom_xpos[geom_id].copy()
            relative_position = center - end_position
            features.extend(relative_position)

        return np.asarray(features, dtype=float)

    def observation(self):
        position, rotation = self._pose()

        clearance_feature = np.clip(
            self._clearance() / self.SAFETY_DISTANCE,
            -1.0,
            3.0,
        )

        observation = np.r_[
            np.sin(self.data.qpos[:6]),
            np.cos(self.data.qpos[:6]),
            np.clip(
                self.data.qvel[:6] / 2.0,
                -1.0,
                1.0,
            ),
            np.clip(
                (self.goal[:3, 3] - position) / 0.50,
                -2.0,
                2.0,
            ),
            rotation_vector(
                rotation,
                self.goal[:3, :3],
            ) / math.pi,
            np.clip(
                self._obstacle_features(position) / 0.50,
                -2.0,
                2.0,
            ),
            clearance_feature,
        ]

        return observation.astype(np.float32)

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Vật cản KHÔNG thay đổi ở đây. Mọi thông tin obstacle được giữ
        # nguyên đúng như scene.xml.
        mujoco.mj_resetData(self.model, self.data)

        if not self._configuration_is_safe(self.start_q):
            raise RuntimeError(
                "The start configuration collides with an obstacle."
            )

        if self.randomize_goal:
            self.goal_q = self._sample_goal_configuration()
            self.goal = forward_kinematics(self.goal_q)
        elif self.fixed_goal_q is not None:
            self.goal_q = self.fixed_goal_q.copy()
            self.goal = forward_kinematics(self.goal_q)

        else:
            raise ValueError(
                "Provide goal_q or set randomize_goal=True."
            )

        self.data.qpos[:6] = self.start_q
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = self.start_q
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.last_command_samples = []

        return self.observation(), self.info(False, False)

    def apply_target(self, q_target):
        """Nội suy lệnh khớp ở 100 Hz và chạy MuJoCo ở 500 Hz."""
        start = self.data.ctrl[:6].copy()
        target = np.clip(
            np.asarray(q_target, dtype=float),
            Q_MIN,
            Q_MAX,
        )

        self.last_command_samples = []
        any_collision = False

        for index in range(1, self.command_steps + 1):
            ratio = index / self.command_steps
            self.data.ctrl[:6] = (
                start + ratio * (target - start)
            )

            for _ in range(self.physics_steps):
                mujoco.mj_step(self.model, self.data)

            distance, angle = self.errors()
            collision = self._collision()
            any_collision |= collision

            self.last_command_samples.append(
                (
                    self.data.qpos[:6].copy(),
                    distance,
                    angle,
                    collision,
                )
            )

        return self.observation(), self.info(
            any_collision,
            False,
        )

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=float),
            -1.0,
            1.0,
        )

        old_distance, old_angle = self.errors()
        position_progress_scale = old_distance / self.POSITION_GATE
        orientation_progress_scale = old_angle / self.ANGLE_GATE

        # Lưu trạng thái an toàn trước khi thử action.
        safe_qpos = self.data.qpos.copy()
        safe_qvel = self.data.qvel.copy()
        safe_ctrl = self.data.ctrl.copy()
        safe_time = self.data.time

        q_target = (
            self.data.qpos[:6]
            + self.MAX_DELTA_Q * action
        )

        _, transition = self.apply_target(q_target)
        self.steps += 1

        collision = transition["collision"]

        # Safety shield: action gây va chạm bị từ chối.
        if collision:
            self.data.qpos[:] = safe_qpos
            self.data.qvel[:] = safe_qvel
            self.data.ctrl[:] = safe_ctrl
            self.data.time = safe_time
            mujoco.mj_forward(self.model, self.data)

        distance, angle = self.errors()
        clearance = self._clearance()

        position_progress = (
            position_progress_scale
            - distance / self.POSITION_GATE
        )
        orientation_progress = (
            orientation_progress_scale
            - angle / self.ANGLE_GATE
        )

        safety_violation = max(
            0.0,
            (
                self.SAFETY_DISTANCE - clearance
            ) / self.SAFETY_DISTANCE,
        )

        # Vị trí là mục tiêu chính. Hướng chỉ là tín hiệu phụ để tạo tư thế
        # thuận lợi trước khi IK hiệu chỉnh chính xác. An toàn được phạt mạnh.
        reward = (
            5.0 * np.clip(position_progress, -1.0, 1.0)
            + 0.1 * np.clip(orientation_progress, -1.0, 1.0)
            - 0.05 * np.clip(
                distance / self.POSITION_GATE,
                0.0,
                10.0,
            )
            - 0.01
            - 2.0 * safety_violation
            - 0.005 * float(np.mean(action**2))
        )

        inside_gate = distance <= self.POSITION_GATE
        success = bool(
            inside_gate and not collision
        )

        if success:
            reward += 20.0

        if collision:
            reward -= 20.0

        terminated = bool(
            self.stop_at_gate and success
        )
        truncated = self.steps >= self.MAX_STEPS

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            self.info(collision, success),
        )

    def info(self, collision, success):
        distance, angle = self.errors()

        return {
            "position_error_m": distance,
            "orientation_error_rad": angle,
            "clearance_m": self._clearance(),
            "collision": bool(collision),
            "sac_gate_reached": bool(success),
            "goal_position": self.goal[:3, 3].copy(),
            "goal_q": self.goal_q.copy(),
        }

    def close(self):
        pass
