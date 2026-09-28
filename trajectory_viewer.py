import time
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from common import ROOT, SCENE_XML, forward_kinematics


SAMPLE_TIME = 0.01
VIDEO_FPS = 50
CAMERA_AZIMUTH = 200
CAMERA_ELEVATION = -20


def load_excel(filename):
    table = pd.read_excel(filename)
    return table[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()


def make_simulation():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    model.actuator_gainprm[:, 0] *= 4
    model.actuator_biasprm[:, 1] *= 4
    model.actuator_biasprm[:, 2] *= 2
    model.actuator_forcerange[:] *= 4
    return model, data


def add_sphere(scene, position, radius, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
        position, np.eye(3).ravel(), color,
    )
    scene.ngeom += 1


def add_axis(scene, origin, direction, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
        origin, origin + 0.14 * direction,
    )
    geom.rgba[:] = color
    scene.ngeom += 1


def add_markers(scene, path):
    start = forward_kinematics(path[0])
    goal = forward_kinematics(path[-1])
    add_sphere(scene, start[:3, 3], 0.040, np.array([0, 1, 0, 1.0]))
    add_axis(scene, start[:3, 3], np.array([0, 0, 1.0]), np.array([0, 1, 0, 1.0]))
    add_sphere(scene, goal[:3, 3], 0.040, np.array([1, 0.85, 0, 1.0]))
    colors = (
        np.array([1, 0, 0, 1.0]),
        np.array([0, 1, 0, 1.0]),
        np.array([0, 0.4, 1, 1.0]),
    )
    for axis, color in zip(goal[:3, :3].T, colors):
        add_axis(scene, goal[:3, 3], axis, color)


def set_start(data, path):
    data.qpos[:6] = path[0]
    data.qvel[:] = 0
    data.ctrl[:6] = path[0]
    data.time = 0


def record_video(path, output):
    model, data = make_simulation()
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    set_start(data, path)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.2, -0.1, 0.35]
    camera.distance = 1.8
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    writer = imageio.get_writer(output, fps=VIDEO_FPS, codec="libx264", quality=8)

    def write_frame():
        renderer.update_scene(data, camera=camera)
        add_markers(renderer.scene, path)
        writer.append_data(renderer.render())

    try:
        for _ in range(2 * VIDEO_FPS):
            write_frame()

        for index, target in enumerate(path[1:], start=1):
            data.ctrl[:6] = target
            for _ in range(5):
                mujoco.mj_step(model, data)
            if index % 2 == 0:
                write_frame()

        data.ctrl[:6] = path[-1]
        for index in range(200):
            for _ in range(5):
                mujoco.mj_step(model, data)
            if index % 2 == 0:
                write_frame()
    finally:
        writer.close()
        renderer.close()


def main():
    default_file = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else ROOT / "results" / "rrt_path.xlsx"
    )
    path = load_excel(default_file)
    model, data = make_simulation()
    set_start(data, path)
    mujoco.mj_forward(model, data)

    state = {
        "path": path, "file": default_file, "index": 1,
        "playing": False, "reset": False, "record": None,
        "markers": True, "new_clock": True, "open": True,
    }

    window = tk.Tk()
    window.title("UR5e trajectory")
    filename = tk.StringVar(value=str(default_file))
    status = tk.StringVar(value="Ready")

    file_row = tk.Frame(window)
    file_row.pack(padx=12, pady=(12, 6))
    tk.Entry(file_row, textvariable=filename, width=62).pack(side="left", padx=(0, 5))

    def browse():
        selected = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if selected:
            filename.set(selected)

    def load():
        selected = Path(filename.get())
        if not selected.is_absolute():
            selected = ROOT / selected
        state.update(
            path=load_excel(selected), file=selected, index=1,
            playing=False, reset=True, markers=True,
        )
        status.set(f"Loaded: {selected.name}")

    tk.Button(file_row, text="Browse", command=browse).pack(side="left", padx=3)
    tk.Button(file_row, text="Load", command=load).pack(side="left", padx=3)

    buttons = tk.Frame(window)
    buttons.pack(padx=12, pady=6)

    def reset():
        state.update(reset=True, playing=False)

    def pause():
        state["playing"] = False
        status.set("Paused")

    def play():
        state.update(playing=True, new_clock=True)
        status.set("Playing")

    def record():
        output = filedialog.asksaveasfilename(
            initialdir=ROOT / "results",
            initialfile=state["file"].stem + "_video.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if output:
            state["record"] = Path(output)

    for text, command in (
        ("Reset", reset), ("Pause", pause),
        ("Play", play), ("Record", record),
    ):
        tk.Button(buttons, text=text, width=10, command=command).pack(side="left", padx=4)

    tk.Label(window, textvariable=status).pack(pady=(0, 10))
    window.protocol("WM_DELETE_WINDOW", lambda: state.update(open=False))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_sample = time.perf_counter()

        while viewer.is_running() and state["open"]:
            window.update_idletasks()
            window.update()

            if state["record"] is not None:
                output = state["record"]
                state["record"] = None
                state["playing"] = False
                status.set("Recording...")
                window.update()
                record_video(state["path"], output)
                status.set(f"Saved: {output.name}")

            if state["reset"]:
                set_start(data, state["path"])
                mujoco.mj_forward(model, data)
                state.update(index=1, reset=False, new_clock=True)
                status.set("Ready")
                viewer.sync()

            if state["markers"]:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_markers(viewer.user_scn, state["path"])
                state["markers"] = False
                viewer.sync()

            if state["new_clock"]:
                next_sample = time.perf_counter()
                state["new_clock"] = False

            now = time.perf_counter()
            if state["playing"] and state["index"] < len(state["path"]) and now >= next_sample:
                data.ctrl[:6] = state["path"][state["index"]]
                for _ in range(5):
                    mujoco.mj_step(model, data)
                state["index"] += 1
                next_sample += SAMPLE_TIME
                status.set(f"Sample {state['index']} / {len(state['path'])}")
                viewer.sync()
                if state["index"] == len(state["path"]):
                    state["playing"] = False
                    status.set("Finished")
            else:
                time.sleep(0.002)

    window.destroy()


if __name__ == "__main__":
    main()
