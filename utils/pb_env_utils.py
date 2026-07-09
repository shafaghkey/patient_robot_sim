"""
Unified PyBullet environment utility.

Usage example:
from utils.path_setup_utils import env
eid = env.connect(gui=True)
env.load_world()
human_id = env.load_urdf("human_gazebo_shkey.urdf", base_pos=[-0.1,1.05,0.1], base_quat=[0,0,-0.707,0.707])
robot_id = env.load_urdf("iiwa14.urdf", use_fixed_base=True)
cup_id   = env.load_mesh_object("objects/plastic_coffee_cup.obj",
                                "objects/plastic_coffee_cup_vhacd.obj",
                                pos=[0.55,-0.15,0.6],
                                euler=[1.5708,-1.5708,0])
for _ in range(1000):
    env.step()
"""
import os, sys, math, time
import numpy as np
import pybullet as pb
import pybullet_data

# ---------- Path discovery ----------
_THIS_FILE = os.path.abspath(__file__)
PKG_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))

_CANDIDATES = [
    os.path.join(PKG_ROOT, "agents"),
]
AGENTS_DIR = next((c for c in _CANDIDATES if os.path.isdir(c)), _CANDIDATES[0])
OBJECTS_DIR = os.path.join(AGENTS_DIR, "objects")

def ensure_python_path():
    if PKG_ROOT not in sys.path:
        sys.path.insert(0, PKG_ROOT)
ensure_python_path()

def asset_path(*rel):
    return os.path.join(AGENTS_DIR, *rel)

def object_path(name):
    return os.path.join(OBJECTS_DIR, name)

def assert_file(path, label="file"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} missing: {path}")
    return path

def configure_pybullet_search_paths(extra_dirs=None, verbose=False):
    # pb.setAdditionalSearchPath keeps only the single most-recently-set path (it does not
    # accumulate a list), so bare-filename loads (e.g. "plane.urdf") need pybullet_data's path
    # to be the last call here. Callers reach agents/objects assets via asset_path()/object_path(),
    # which already return absolute paths and don't depend on this search path at all.
    for d in (AGENTS_DIR, OBJECTS_DIR):
        if not os.path.isdir(d) and verbose:
            print(f"[WARN] Missing dir: {d}")
    if extra_dirs:
        for d in extra_dirs:
            if not os.path.isdir(d) and verbose:
                print(f"[WARN] Extra missing: {d}")
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    if verbose:
        print(f"[INFO] Search paths set. AGENTS_DIR={AGENTS_DIR}")

# ---------- Environment class ----------
class PyBulletEnv:
    def __init__(self):
        self.cid = None
        self.gui = False
        self.time_step = 1/1000.0
        self.connected_time = None
        self.loaded_plane = False

    # --- Connection / setup ---
    def connect(self, gui: bool = True, options: str = ""):
        """
        Connect to PyBullet (GUI or DIRECT mode). If already connected, does nothing.
        Args:
            - gui: True -> pb.GUI, False -> pb.DIRECT
            - options: connection options string (see pybullet docs)
        """
        if self.cid is not None and pb.getConnectionInfo(self.cid)["isConnected"]:
            return self.cid
        self.gui = gui
        mode = pb.GUI if gui else pb.DIRECT
        try:
            self.cid = pb.connect(mode, options=options)  # may raise
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            return None
        pb.resetSimulation()
        configure_pybullet_search_paths()
        # pb.setPhysicsEngineParameter(numSolverIterations=120)
        # Make contacts softer and solver tighter (reduces impulses/oscillation)
        pb.setPhysicsEngineParameter(numSolverIterations=200, erp=0.2, contactERP=0.2, frictionERP=0.2)
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(self.time_step)
        if self.gui:
            # pb.resetDebugVisualizerCamera(2.0, 60, -30, [0,0,0.9])
            pb.resetDebugVisualizerCamera(1, 70, -10, [0.1,-0.5,0.7])
            self.apply_debug_viz(gui_panels=False, shadows=False, previews=False)
        self.connected_time = time.time()
        self.loaded_plane = False
        return self.cid
    
    def apply_debug_viz(self, gui_panels: bool = False, shadows: bool = False, previews: bool = False):
        """
        Toggle GUI panels, shadows, and buffer previews (GUI mode only).
        """
        if not self.gui:
            return
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, int(gui_panels)) # Show/hide the on-screen GUI panels (sliders, buttons). Frees screen space and a bit of CPU/GPU.
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, int(shadows))    # Enable/disable shadow rendering for objects. Improves visuals, costs GPU.
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, int(previews))    # Enable/disable the RGB camera buffer preview overlay windows.
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, int(previews))  # Enable/disable the depth buffer preview overlay windows.
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, int(previews)) # Enable/disable the segmentation buffer preview overlay windows.

    def disconnect(self):
        if self.cid is not None:
            try:
                pb.disconnect(self.cid)
            except Exception:
                pass
        self.cid = None

    def set_time_step(self, dt):
        self.time_step = float(dt)
        pb.setTimeStep(self.time_step)

    # --- World loading ---
    def load_world(self, plane=True):
        if plane and not self.loaded_plane:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.loadURDF("plane.urdf", [0,0,0], useFixedBase=True)
            self.loaded_plane = True

    # --- Asset loading ---
    def load_urdf(self, urdf_name, base_pos=(0,0,0), base_quat=(0,0,0,1),
                  use_fixed_base=True, flags=pb.URDF_USE_INERTIA_FROM_FILE): #flags=pb.URDF_USE_SELF_COLLISION
        path = asset_path(urdf_name)
        print(f"[DEBUG] Loading URDF from: {path}")
        assert_file(path, "URDF")
        return pb.loadURDF(path, base_pos, base_quat, useFixedBase=use_fixed_base, flags=flags)

    def load_mesh_object(self, visual_rel, collision_rel=None,
                         pos=(0,0,0), euler=None, quat=None, mass=0.0,
                         rgba=(1,0.5,1,0.9), scale=(0.05,0.05,0.05)):
        visual_file = object_path(visual_rel)
        assert_file(visual_file, "visual mesh")
        if collision_rel:
            collision_file = object_path(collision_rel)
            assert_file(collision_file, "collision mesh")
        else:
            collision_file = visual_file
        if euler and not quat:
            quat = pb.getQuaternionFromEuler(euler)
        quat = quat or (0,0,0,1)
        vs = pb.createVisualShape(pb.GEOM_MESH, fileName=visual_file, meshScale=scale, rgbaColor=rgba)
        cs = pb.createCollisionShape(pb.GEOM_MESH, fileName=collision_file, meshScale=scale)
        bid = pb.createMultiBody(baseMass=mass,         # Zero mass makes it immovable
                                 baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                                 basePosition=pos, baseOrientation=quat,
                                #  useMaximalCoordinates=True,  # Better stability for static objects 
                                )
        return bid

    
    # --- Simulation ---
    def step(self, n=1):
        """Step the simulation n times (default=1)."""
        for _ in range(n):
            pb.stepSimulation()

    def pause(self):
        try:
            while True:
                env.step()
                time.sleep(env.time_step)
        except KeyboardInterrupt:
            pass

    # # --- Utility ---
    # def clamp(self, arr, lim):
    #     arr = np.array(arr, dtype=float)
    #     return np.clip(arr, -abs(lim), abs(lim))
    
    # # --- Control helpers ---
    # def disable_default_motors(self, body_id, joint_indices=None):
    #     if joint_indices is None:
    #         joint_indices = [j for j in range(pb.getNumJoints(body_id))
    #                          if pb.getJointInfo(body_id, j)[2] != pb.JOINT_FIXED]
    #     pb.setJointMotorControlArray(body_id, joint_indices,
    #                                  controlMode=pb.VELOCITY_CONTROL,
    #                                  forces=[0.0]*len(joint_indices))
    #     return joint_indices

    # def apply_cmd_torque(self, body_id, joint_indices, torques):
    #     torques = np.array(torques, dtype=float).flatten().tolist()
    #     pb.setJointMotorControlArray(body_id, joint_indices, 
    #                                  pb.TORQUE_CONTROL, 
    #                                  forces=torques,
    #                                  positionGains=[0.0]*len(joint_indices),    # disable pos ctrl
    #                                  velocityGains=[0.0]*len(joint_indices),    # disable vel ctrl
    #                                 )

    # # --- State queries ---
    # def joint_states(self, body_id, joint_indices=None):
    #     if joint_indices is None:
    #         joint_indices = list(range(pb.getNumJoints(body_id)))
    #     js = pb.getJointStates(body_id, joint_indices)
    #     q = [s[0] for s in js]
    #     dq = [s[1] for s in js]
    #     trq = [s[3] for s in js]
    #     return q, dq, trq

    # def link_state(self, body_id, link_index, vel=True):
    #     return pb.getLinkState(body_id, link_index, computeLinkVelocity=int(vel))

    # def jacobian(self, body_id, link_index, joint_positions=None):
    #     if joint_positions is None:
    #         joint_positions, _, _ = self.joint_states(body_id)
    #     link_state = pb.getLinkState(body_id, link_index)
    #     local_pos = link_state[2]
    #     J_t, J_r = pb.calculateJacobian(body_id, link_index, local_pos,
    #                                     joint_positions,
    #                                     [0.0]*len(joint_positions),
    #                                     [0.0]*len(joint_positions))
    #     return np.array(J_t), np.array(J_r)

# Singleton instance
env = PyBulletEnv()

# If executed directly, do a quick self-test
if __name__ == "__main__":
    env.connect(gui=True)
    env.load_world()
    env.load_urdf("table_human_stool.urdf", use_fixed_base=True)
    print("[TEST] AGENTS_DIR:", AGENTS_DIR)
    print("[TEST] Plane loaded OK.")
    # kuka_base = [0.5, 0, 0]
    # robot = env.load_urdf("iiwa14_gripper.urdf", use_fixed_base=True)
    # print("[TEST] Robot loaded OK.")
    # stay connected:
    try:
        while True:
            env.step()
            time.sleep(env.time_step)
    except KeyboardInterrupt:
        pass




    
