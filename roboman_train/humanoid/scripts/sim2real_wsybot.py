import math
import time
import numpy as np
import serial
import os
import time
import struct

# ====
import threading
import queue

# ====
from isaacgym import gymapi
import torch
from tqdm import tqdm
from collections import deque

from humanoid.envs.custom.acuator.imu.imu import IMU
from humanoid.envs.custom.wsybot_config import wsybotCfg as Cfg


class Sim2RealController:
    def __init__(self):
        # Constants
        self.motor_lower_limit = [
            -0.15,
            -1.0,
            -1.0,
            -0.4,
            -1.0,
            -0.5,
            -0.7,
            -1.2,
            -1.0,
            -1.5,
            -1.0,
            -0.5,
        ]
        self.motor_higher_limit = [
            0.7,
            1.2,
            1.0,
            1.5,
            1.0,
            0.5,
            0.15,
            1.0,
            1.0,
            0.4,
            1.0,
            0.5,
        ]

        # Command class
        class Cmd:
            vx = 0.0
            vy = 0.0
            dyaw = 0.0

        self.cmd = Cmd()

        # Initialize IMU and serial communication
        os.system("sudo chmod a+rw /dev/ttyUSB0")
        self.imu_port = "/dev/ttyUSB0"
        self.imu_baudrate = 115200
        self.imu = IMU()

        os.system("sudo chmod a+rw /dev/ttyACM0")
        self.control_port = "/dev/ttyACM0"
        self.control_baudrate = 115200
        self.ser = serial.Serial("/dev/ttyACM0", 115200, timeout=0)
        self.ser.timeout = 0.005
        self.ser.reset_input_buffer()

        # Load policy
        self.policy_file_path = "/home/wsy/Desktop/humanoid-gym-main/logs/wsybot_v.3_ppo/exported/policies/policy_1.pt"
        self.policy = torch.jit.load(
            self.policy_file_path, map_location=torch.device("cpu")
        )

        # Queues for IMU and motor data
        self.imu_queue = queue.Queue(maxsize=1)
        self.motor_queue = queue.Queue(maxsize=1)

        # Initialize variables
        self.target_q_int16 = [32768] * 12
        self.target_q = [0.0] * 12
        self.obs = np.zeros([1, 47], dtype=np.float32)
        self.count_lowlevel = 0
        self.hist_obs = deque()
        self.action = np.zeros((12), dtype=np.double)
        self.policy_input = np.zeros([1, 705], dtype=np.float32)

        # Initialize motor utilities
        self.motor_utils = MotorUtils()

        # Initialize threads
        self.imu_thread = threading.Thread(target=self.read_imu_data)
        self.motor_thread = threading.Thread(target=self.control_motor)

    def quaternion_to_euler(self, quat):
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        return np.array([roll_x, pitch_y, yaw_z])

    def read_imu_data(self):
        while True:
            try:
                quat, omega = self.imu.cmd_read(self.imu_port, self.imu_baudrate)
                if self.imu_queue.full():
                    self.imu_queue.get()
                self.imu_queue.put((quat, omega))
            except Exception as e:
                print(f"Error reading IMU data: {e}")
            time.sleep(0.001)

    def control_motor(self):
        while True:
            try:
                buffer = self.ser.read(50)
                if len(buffer) == 50:
                    p_float, v_float = self.motor_utils.decode_motor_data(
                        buffer, -12.5, 12.5, -30.0, 30.0, 16
                    )
                    if self.motor_queue.full():
                        self.motor_queue.get()
                    self.motor_queue.put((p_float, v_float))
            except Exception as e:
                print(f"Error reading motor data: {e}")
            time.sleep(0.001)

    def run(self):
        self.imu_thread.start()
        self.motor_thread.start()

        for i in range(12):
            self.target_q_int16[i] = self.motor_utils.float_to_uint(
                self.target_q[i], -12.5, 12.5, 16
            )
        self.motor_utils.execute(self.ser, self.target_q_int16)
        time.sleep(2)

        for _ in range(15):
            self.hist_obs.append(np.zeros([1, 47], dtype=np.double))

        while True:
            start_time = time.time()
            if self.count_lowlevel > 0:
                try:
                    quat, omega = self.imu_queue.get_nowait()
                except queue.Empty:
                    continue
                time.sleep(0.001)
                try:
                    p_float, v_float = self.motor_queue.get_nowait()
                except queue.Empty:
                    continue

                flag_time = time.time()
                if flag_time - start_time < 0.015:
                    pass
                else:
                    continue

                time.sleep(0.002)
                eu_ang = self.quaternion_to_euler(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                self.obs[0, 0] = math.sin(
                    2 * math.pi * self.count_lowlevel * 0.01 / 0.70
                )
                self.obs[0, 1] = math.cos(
                    2 * math.pi * self.count_lowlevel * 0.01 / 0.70
                )
                self.obs[0, 2] = self.cmd.vx * 2
                self.obs[0, 3] = self.cmd.vy * 2
                self.obs[0, 4] = self.cmd.dyaw * 1
                self.obs[0, 5:17] = np.array(p_float) * 1
                self.obs[0, 17:29] = np.array(v_float) * 0.05
                self.obs[0, 29:41] = self.action
                self.obs[0, 41:44] = omega
                self.obs[0, 44:47] = eu_ang

                self.obs = np.clip(self.obs, -18, 18)
                self.hist_obs.append(self.obs)
                self.hist_obs.popleft()

                for i in range(15):
                    self.policy_input[0, i * 47 : (i + 1) * 47] = self.hist_obs[i][0, :]

                policy_input_tensor = torch.from_numpy(self.policy_input)
                self.action[:] = self.policy(policy_input_tensor)[0].detach().numpy()
                self.action = np.clip(self.action, -18, 18)

                self.target_q = self.action * 0.3
                self.target_q = np.clip(
                    self.target_q, self.motor_lower_limit, self.motor_higher_limit
                )

                for i in range(12):
                    self.target_q_int16[i] = self.motor_utils.float_to_uint(
                        self.target_q[i], -12.5, 12.5, 16
                    )

                self.motor_utils.execute(self.ser, self.target_q_int16)

            self.count_lowlevel += 1
            execute_time = time.time() - start_time
            if execute_time < 0.0197:
                time.sleep(0.0197 - execute_time)
            else:
                print("SingleFilm :", execute_time)


if __name__ == "__main__":
    controller = Sim2RealController()
    controller.run()
