import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class RelationalPickAndPlaceEnv(gym.Env):
    def __init__(self, render=False):
        super(RelationalPickAndPlaceEnv, self).__init__()
        self.render = render
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.timeStep = 1./240.
        p.setTimeStep(self.timeStep)

        self.plane = p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])

        # Load Franka Panda arm URDF
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # Objects: a big cube and a small cube
        self.big_cube = p.loadURDF("cube_small.urdf", basePosition=[0.65, 0, 0.63])
        self.small_cube = p.loadURDF("cube_small.urdf", basePosition=[0.55, -0.1, 0.63])

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1, 1, 1, 1, 1], dtype=np.float32)
        )

        # Observation: 7 joint positions + 6 object positions + 1 gripper pos = 14 dims
        obs_low = np.array([-np.pi]*7 + [-10]*6 + [0], dtype=np.float32)
        obs_high = np.array([np.pi]*7 + [10]*6 + [0.1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.max_steps = 200
        self.current_step = 0

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timeStep)
        self.plane = p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.big_cube = p.loadURDF("cube_small.urdf", basePosition=[0.65, 0, 0.63])
        self.small_cube = p.loadURDF("cube_small.urdf", basePosition=[0.55, -0.1, 0.63])

        self.current_step = 0

        # Reset robot joints to initial positions
        initial_joint_positions = [0, -0.4, 0, -2.3, 0, 2, 0.8]
        for i in range(7):
            p.resetJointState(self.robot, i, initial_joint_positions[i])

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        action = np.array of shape (8,)
        [dx, dy, dz, qx, qy, qz, qw, grip]
        """

        # Apply action - update end effector position and orientation
        current_pose = p.getLinkState(self.robot, 11)  # panda_hand is link 11
        current_pos = np.array(current_pose[4])  # world position
        current_orient = np.array(current_pose[5])  # quaternion

        delta_pos = np.clip(action[:3], -0.05, 0.05)
        new_pos = current_pos + delta_pos

        new_orient = action[3:7]
        new_orient = new_orient / np.linalg.norm(new_orient)  # normalize quaternion

        # Inverse kinematics for new pose
        joint_poses = p.calculateInverseKinematics(
            self.robot, 11, new_pos, new_orient,
            maxNumIterations=20
        )

        # Set joint motors to the IK results for 7 joints + fingers
        for i in range(7):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, joint_poses[i], force=200)

        # Grip control: simple open/close fingers
        grip_pos = np.clip(action[7], 0, 1)
        finger_pos = grip_pos * 0.04  # max finger opening approx 0.04
        p.setJointMotorControl2(self.robot, 9, p.POSITION_CONTROL, finger_pos, force=50)
        p.setJointMotorControl2(self.robot, 10, p.POSITION_CONTROL, finger_pos, force=50)

        p.stepSimulation()
        if self.render:
            time.sleep(self.timeStep)

        obs = self._get_obs()
        reward = self._compute_reward()

        terminated = self._check_done()
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Get joint states
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(7)]

        # Get object positions
        big_pos, _ = p.getBasePositionAndOrientation(self.big_cube)
        small_pos, _ = p.getBasePositionAndOrientation(self.small_cube)

        # Gripper opening
        finger_pos = p.getJointState(self.robot, 9)[0]

        obs = np.array(joint_states + list(big_pos) + list(small_pos) + [finger_pos], dtype=np.float32)
        return obs

    def _compute_reward(self):
        # Reward = 1 if small cube is stacked on big cube (z-distance small)
        small_pos, _ = p.getBasePositionAndOrientation(self.small_cube)
        big_pos, _ = p.getBasePositionAndOrientation(self.big_cube)

        dist_xy = np.linalg.norm(np.array(small_pos[:2]) - np.array(big_pos[:2]))
        dist_z = abs(small_pos[2] - (big_pos[2] + 0.05))  # 0.05 ~ cube height

        if dist_xy < 0.03 and dist_z < 0.03:
            return 1.0
        else:
            return -dist_xy - dist_z  # encourage getting closer

    def _check_done(self):
        reward = self._compute_reward()
        return reward >= 1.0

    def close(self):
        p.disconnect()
