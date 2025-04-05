import time
import matplotlib.pyplot as plt

# from sim2real.module.imu.imu import IMU
from humanoid.envs.custom.acuator.imu.imu import IMU

# imu init
# If there are multiple USB devices here, replace them with the actual serial port names.
import os

os.system("sudo chmod a+rw /dev/ttyUSB0")
port = "/dev/ttyUSB0"
baudrate = 115200
imu = IMU(port, baudrate)

# 记录 quat 和 omega 的数据
quat_data = []
omega_data = []

# 读取数据的次数
num_readings = 50  # 可以根据需要调整读取次数

# 循环读取数据
for _ in range(num_readings):
    try:
        quat, omega = imu.cmd_read()
        if quat is not None and omega is not None:
            quat_data.append(quat)
            omega_data.append(omega)
            print(f"Quaternion: {quat}, Angular Velocity: {omega}")
        else:
            print("Failed to read valid IMU data. Skipping...")
    except Exception as e:
        print(f"An error occurred while reading IMU data: {e}")
    time.sleep(0.1)  # 可以根据需要调整读取间隔

# 检查是否有有效的数据
if quat_data and omega_data:
    # 将数据转换为适合绘图的格式
    quat_data = list(map(list, zip(*quat_data)))
    omega_data = list(map(list, zip(*omega_data)))

    # 创建一个包含两个子图的图形
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 绘制 quat 数据
    labels_quat = ["w", "x", "y", "z"]
    for i, data in enumerate(quat_data):
        axes[0].plot(data, label=labels_quat[i])
    axes[0].set_title("Quaternion Data")
    axes[0].set_xlabel("Reading Number")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    # 绘制 omega 数据
    labels_omega = ["x", "y", "z"]
    for i, data in enumerate(omega_data):
        axes[1].plot(data, label=labels_omega[i])
    axes[1].set_title("Angular Velocity Data")
    axes[1].set_xlabel("Reading Number")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()
else:
    print("No valid IMU data was collected. Cannot generate plot.")
