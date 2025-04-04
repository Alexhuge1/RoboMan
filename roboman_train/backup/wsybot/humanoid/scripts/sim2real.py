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

import matplotlib.pyplot as plt

# 定义常量
DATASIZE = 44
StartByte = 0xAB
EndByte = 0xBA
# pos limit order=MOTOR 1     2     3     4     5     6     7     8     9     A     B     C
motor_lower_limit = [
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
]
motor_higher_limit = [
    1.0, 
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
]

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

# #imu decoding
def quaternion_to_euler(quat):
    # 计算偏航角（yaw）
    w, x, y, z = quat
    # x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians

    # print(yaw_z)
    return np.array([roll_x, pitch_y, yaw_z])

def decode_motor_data(data, p_min, p_max, v_min, v_max, bits):
    # 检查数据长度是否正确（假设数据长度固定为50字节，包括帧头和帧尾）
    if len(data) != 50:
        raise ValueError("Data length is incorrect")
    # 验证帧头和帧尾
    if data[0] != 0xAB or data[-1] != 0xBA:
        raise ValueError("Frame header or footer is incorrect")
    # 初始化存储p_float和v_float的列表
    p_float_list = []
    v_float_list = []
    # 解码p_int值并转换为p_float
    for i in range(12):  #
        # 计算p_int的字节索引（跳过帧头，每个p_int占2个字节）
        byte_index = 1 + 2 * i
        # 提取p_int的高字节和低字节，并组合成一个整数
        p_int = (data[byte_index] << 8) + data[byte_index + 1]
        # 使用uint_to_float函数将p_int转换为p_float
        p_float = uint_to_float(p_int, p_min, p_max, bits)
        p_float_list.append(p_float)
    # 解码v_int值并转换为v_float（与p_int类似）
    for i in range(12):  #
        # 计算v_int的字节索引（跳过帧头和p_int数据，每个v_int占2个字节）
        byte_index = 25 + 2 * i  # 25是p_int数据结束后的起始索引
        # 提取v_int的高字节和低字节，并组合成一个整数
        v_int = (data[byte_index] << 8) + data[byte_index + 1]
        # 使用uint_to_float函数将v_int转换为v_float（注意这里仍然使用uint_to_float，只是参数不同）
        v_float = uint_to_float(v_int, v_min, v_max, bits)
        v_float_list.append(v_float)

    return p_float_list, v_float_list
#解码是否正确

# uint to float
def uint_to_float(x_int, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    return (x_int * span) / ((1 << bits) - 1) + offset


# float to uint
def float_to_uint(x_float, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    return int((x_float - offset) * ((1 << bits) - 1) / span)


# print to test
def print_array_data(data_array):
    # 定义数据字段的名称
    data_labels = [
        "Motor_Pos  1",
        "Motor_Pos  2",
        "Motor_Pos  3",
        "Motor_Pos  4",
        "Motor_Pos  5",
        "Motor_Pos  6",
        "Motor_Pos  7",
        "Motor_Pos  8",
        "Motor_Pos  9",
        "Motor_Pos 10",
        "Motor_Pos 11",
        "Motor_Pos 12",
    ]
    # 检查输入数组的长度是否正确
    if len(data_array) != len(data_labels):
        raise ValueError(
            f"Expected {len(data_labels)} elements in the data array, but got {len(data_array)} elements."
        )
    # 打印数据
    for label, value in zip(data_labels, data_array):
        print(f"{label:<12}: {value:<9.4f}")

# print to test
def print_array_data2(data_array):
    # 定义数据字段的名称
    data_labels = [
        "Motor_Vel  1",
        "Motor_Vel  2",
        "Motor_Vel  3",
        "Motor_Vel  4",
        "Motor_Vel  5",
        "Motor_Vel  6",
        "Motor_Vel  7",
        "Motor_Vel  8",
        "Motor_Vel  9",
        "Motor_Vel 10",
        "Motor_Vel 11",
        "Motor_Vel 12",
    ]
    # 检查输入数组的长度是否正确
    if len(data_array) != len(data_labels):
        raise ValueError(
            f"Expected {len(data_labels)} elements in the data array, but got {len(data_array)} elements."
        )
    # 打印数据
    for label, value in zip(data_labels, data_array):
        print(f"{label:<12}: {value:<9.4f}")


os.system("sudo chmod a+rw /dev/ttyUSB0")
imu_port = "/dev/ttyUSB0"
imu_baudrate = 115200
imu = IMU()
os.system("sudo chmod a+rw /dev/ttyACM0")
control_port = "/dev/ttyACM0"
contro_baudrate = 115200
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=0)
ser.timeout = 0.001
ser.reset_input_buffer()  # 清空输入缓冲区

policy_file_path = (
    "/home/wsy/Desktop/humanoid-gym-main/logs/wsybot_ppo/exported/policies/policy_1.pt"
)

# policy_file_path = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "policy_yyx.pt",
# )

policy = torch.jit.load(policy_file_path, map_location=torch.device("cpu"))
# ====
# 创建队列用于存储 IMU 数据和电机数据
imu_queue = queue.Queue(maxsize=1)
motor_queue = queue.Queue(maxsize=1)


# ==== threading time decline processes
def read_imu_data():
    global imu_queue
    while True:
        try:
            quat, omega = imu.cmd_read(imu_port, imu_baudrate)
            if imu_queue.full():
                # 如果队列满了，直接丢弃旧数据
                imu_queue.get()
            imu_queue.put((quat, omega))
        except Exception as e:
            print(f"Error reading IMU data: {e}")
            time.sleep(0.0005)  # 避免频繁报错
        # time.sleep(0.001)


def control_motor():
    global motor_queue, ser
    while True:
        try:
            buffer = ser.read(50)
            if len(buffer) == 50:
                p_float, v_float = decode_motor_data(
                    buffer, -12.5, 12.5, -30.0, 30.0, 16
                )
                if motor_queue.full():
                    # 如果队列满了，直接丢弃旧数据
                    motor_queue.get()
                motor_queue.put((p_float, v_float))
        except Exception as e:
            print(f"Error reading motor data: {e}")
            time.sleep(0.0005)  # 避免频繁报错
        # time.sleep(0.001)


# 创建并启动线程
imu_thread = threading.Thread(target=read_imu_data)
motor_thread = threading.Thread(target=control_motor)

imu_thread.start()
motor_thread.start()


# ====execution start
def execute(target_q):
    packed_target_q = struct.pack("<12H", *target_q)
    packed_data = bytes([StartByte]) + packed_target_q + bytes([EndByte])
    if not ser.is_open:
        ser.open
    ser.write(packed_data)


# ====execution end

target_q_int16 = [32768] * 12
target_q = [0.0] * 12

for i in range(12):
    target_q_int16[i] = float_to_uint(target_q[i], -12.5, 12.5, 16)
execute(target_q_int16)
time.sleep(2)

obs = np.zeros([1, 47], dtype=np.float32)
count_lowlevel = 0

hist_obs = deque()
time.sleep(0.001)

for _ in range(15):

    hist_obs.append(np.zeros([1, 47], dtype=np.double))
    # hist_obs.append(hist_obs + np.random.normal(0,0.01,47))

# hist_obs = deque()
# initial_obs = np.zeros((1, 47), dtype=np.double)  # 创建初始观测模板

# for _ in range(15):
#     # 生成与观测维度匹配的噪声 (1,47)
#     noise = np.random.normal(0, 0.01, (1, 47))
#     # 添加带噪声的初始观测
#     hist_obs.append(initial_obs + noise)

action = np.zeros((12), dtype=np.double)

policy_input = np.zeros([1, 705], dtype=np.float32)

p_float_list = []
target_q_list = []

while True:
    # if count_lowlevel < 500:
    start_time = time.time()
    # if count_lowlevel > 0:
    # start_time = time.time()

    time.sleep(0.0005)
    try:
        p_float, v_float = motor_queue.get_nowait()
        # p_float, v_float = motor_queue.get(timeout=0.005)
    except queue.Empty:
        print("Motor fail")
        print(count_lowlevel * 0.01)
        # continue  # 若队列为空，跳过本次循环

    # ====
    time.sleep(0.0005)

    if count_lowlevel % 1 == 0:
        # 非阻塞方式获取数据
        try:
            quat, omega = imu_queue.get_nowait()
            # quat, omega = imu_queue.get(timeout=0.005)
        except queue.Empty:
            print("IMU fail")
            print(count_lowlevel * 0.01)
            # continue  # 若队列为空，跳过本次循环
        eu_ang = quaternion_to_euler(quat)

        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        # print(eu_ang)
        omega = [0.0, 0.0, 0.0]
        eu_ang = [0.0, 0.0, 0.0]

        obs[0, 0] = math.cos(2 * math.pi * count_lowlevel * 0.02 / 0.70)
        obs[0, 1] = math.sin(2 * math.pi * count_lowlevel * 0.02 / 0.70)
        obs[0, 2] = cmd.vx * 2
        obs[0, 3] = cmd.vy * 2
        obs[0, 4] = cmd.dyaw * 1
        obs[0, 5:17] = np.array(p_float) * 1
        obs[0, 17:29] = np.array(v_float) * 0.05
        obs[0, 29:41] = action
        obs[0, 41:44] = omega
        obs[0, 44:47] = eu_ang

        obs = np.clip(obs, -18, 18)
        print('obs[0, 0] ', obs[0, 0] )
        print('obs[0, 1] ', obs[0, 1] )
        print('obs[0, 2] ', obs[0, 2] )
        print('obs[0, 3] ', obs[0, 3] )
        print('obs[0, 4] ', obs[0, 4] )
        print('obs[0, 5:17]', obs[0, 5:17])
        print('obs[0, 17:29]', obs[0, 17:29])
        print('obs[0, 29:41]', obs[0, 29:41])
        print('obs[0, 41:44]', obs[0, 41:44])
        print('obs[0, 44:47]', obs[0, 44:47])

        hist_obs.append(obs)
        hist_obs.popleft()

        for i in range(15):
            policy_input[0, i * 47 : (i + 1) * 47] = hist_obs[i][0, :]

        policy_input_tensor = torch.tensor(policy_input)
        action[:] = policy(policy_input_tensor)[0].detach().numpy()
        action = np.clip(action, -18, 18)

        target_q = action * 0.3
        target_q = np.clip(target_q, motor_lower_limit, motor_higher_limit)

        # target_q = [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # print_array_data(target_q)
        for i in range(12):
            target_q_int16[i] = float_to_uint(target_q[i], -12.5, 12.5, 16)
            # target_q_int16[i] = float_to_uint(0.0, -12.5, 12.5, 16)
    execute(target_q_int16)

    time.sleep(0.002)
    execute_falg = time.time()

    p_float_list.append(p_float)
    target_q_list.append(target_q)
    # print(execute_falg - start_time)

    count_lowlevel += 1

    execte_end_time = time.time()
    execute_time = execte_end_time - start_time
    if execute_time < 0.01:
        time.sleep(0.01 - execute_time)
    else:
        print("SingleFilm :", execute_time)
