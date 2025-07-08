import re

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')

# 选择的点数
n = 5

# --- Step 1: 文件选择 ---
root = tk.Tk()
root.withdraw()  # 隐藏主窗口，仅使用文件对话框

# 弹出文件选择对话框，选择多个图像文件
file_paths = filedialog.askopenfilenames(
    title="Select image files",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
)

if not file_paths:
    print("No files selected.")
    exit()

# --- Step 1.5: 读取最大光强值文件 ---
max_i_path = r"C:\\Users\\joengzaang\\Desktop\\课设\\max_i.txt"
with open(max_i_path, encoding='utf-8') as f:
    lines = f.readlines()[1:]  # 跳过表头
    max_i_dict = {}
    for line in lines:
        parts = re.split(r"\s+", line.strip())
        if len(parts) == 2:
            max_i_dict[parts[0]] = float(parts[1])

# --- Step 2: Let user select origin on each image ---
location_file = os.path.join(os.path.dirname(file_paths[0]), "locations.txt")
origin_points = []

location_dict = {}

if os.path.exists(location_file):
    print("📂 从 locations.txt 读取原点坐标")
    with open(location_file, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                fname, x, y = parts
                location_dict[fname] = (int(x), int(y))
else:
    print("🖱 第一次运行，请点击每张图像的原点 (0,0)")
    for path in file_paths:
        img = Image.open(path).convert("L")
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Click origin for: {os.path.basename(path)}")
        coords = plt.ginput(1, timeout=-1)
        plt.close()
        if not coords:
            exit("❌ 未选中原点")
        x, y = int(coords[0][0]), int(coords[0][1])
        fname = os.path.basename(path)
        location_dict[fname] = (x, y)

    # 写入 locations.txt
    with open(location_file, "w", encoding='utf-8') as f:
        for fname, (x, y) in location_dict.items():
            f.write(f"{fname} {x} {y}\n")
    print(f"✅ 原点坐标已写入 {location_file}")

# 将 origin_points 按当前文件顺序重建
for path in file_paths:
    fname = os.path.basename(path)
    if fname in location_dict:
        origin_points.append(location_dict[fname])
    else:
        exit(f"❌ 无法找到图像 {fname} 的原点记录，请删除 locations.txt 重新标注。")


# --- Step 3: 在第一张图片上选取10个参考点 ---
print("🖱 Click 10 reference points in the FIRST image")
first_img = Image.open(file_paths[0]).convert("L")
fig, ax = plt.subplots()
ax.imshow(first_img, cmap='gray')
ax.set_title("Click 10 reference points")
ref_points = []
for i in range(n):
    pt = plt.ginput(1, timeout=-1)[0]
    ax.plot(pt[0], pt[1], 'ro')
    ax.text(pt[0] + 5, pt[1], str(i + 1), color='yellow', fontsize=12)
    fig.canvas.draw()
    ref_points.append(pt)
# ✅ 保留窗口，同时延迟继续执行
plt.pause(0.3)  # 延迟 1 秒再继续执行后续代码（窗口继续显示）


if len(ref_points) != n:
    exit("Must click exactly 3 reference points")

# Convert to relative coordinates
ref_origin = origin_points[0]
relative_coords = [(int(x - ref_origin[0]), int(y - ref_origin[1])) for (x, y) in ref_points]

# --- Step 4: Initialize result list ---
results = []

# --- Step 5: Extract grayscale values for each point and image ---
point_labels = [f"p{i + 1}" for i in range(n)]
gray_labels = ["9045", "090", "045", "00"]  # Ordered for priority
suffix_to_gray = {"00": "gray(0,0)", "090": "gray(0,90)", "045": "gray(0,45)", "9045": "gray(90,45)"}

# Build gray value table: 3 rows (p1~p3), each with 4 gray values
gray_table = {label: {} for label in point_labels}

for path, origin in zip(file_paths, origin_points):
    img = Image.open(path).convert("L")
    filename = os.path.splitext(os.path.basename(path))[0]
    max_i = max_i_dict.get(filename, 1.0)  # 获取对应的最大光强值，默认为1.0
    print(f"✅ 获取光强 {max_i} 用于图像 {filename}")

    # 根据文件名后缀确定灰度角度名称
    gray_name = None
    for suffix in gray_labels:
        if filename.endswith(suffix):
            gray_name = suffix_to_gray[suffix]
            break

    if gray_name is None:
        print(f"⚠️ File {filename} does not match expected suffix (9045, 090, 045, 00)")
        continue

    # 对每个参考点，提取其在该图像中的灰度值
    for i, label in enumerate(point_labels):
        dx, dy = relative_coords[i]
        x, y = origin[0] + dx, origin[1] + dy
        gray_val = img.getpixel((x, y)) if 0 <= x < img.width and 0 <= y < img.height else None
        if gray_val is not None:
            gray_val *= max_i  # 乘以最大光强值
        gray_table[label][gray_name] = gray_val

# --- Step 6: Compute S0~S3 for each point ---
for i, label in enumerate(point_labels):
    dx, dy = relative_coords[i]
    row = {
        "Index": i + 1,
        "Coordinate": f"({dx},{dy})",
        "gray(0,0)": gray_table[label].get("gray(0,0)"),
        "gray(0,90)": gray_table[label].get("gray(0,90)"),
        "gray(0,45)": gray_table[label].get("gray(0,45)"),
        "gray(90,45)": gray_table[label].get("gray(90,45)")
    }
    g00, g90, g45, g9045 = row["gray(0,0)"], row["gray(0,90)"], row["gray(0,45)"], row["gray(90,45)"]

    s0 = g00 + g90 if None not in (g00, g90) else None
    s1 = g00 - g90 if None not in (g00, g90) else None
    s2 = 2 * g45 - s0 if None not in (g45, s0) else None
    s3 = s0 - 2 * g9045 if None not in (g9045, s0) else None

    row["S0"] = s0
    row["S1"] = s1
    row["S2"] = s2
    row["S3"] = s3
    results.append(row)

# --- Step 7: Normalize S0~S3 ---
s_data = pd.DataFrame(results)
s_data["S0*"] = 1.00
for s in ["S1", "S2", "S3"]:
    s_data[s + "*"] = s_data.apply(
        lambda row: round(row[s] / row["S0"], 2) if pd.notna(row[s]) and pd.notna(row["S0"]) and row[
            "S0"] != 0 else None,
        axis=1
    )

# --- Step 8: Compute Degree of Polarization ---
s_data["p"] = s_data.apply(
    lambda row: round(np.sqrt(sum(
        (row[s + "*"] ** 2 for s in ["S1", "S2", "S3"] if pd.notna(row[s + "*"])
         )), ), 2) if all(pd.notna(row[s + "*"]) for s in ["S1", "S2", "S3"]) else None,
    axis=1
)


# --- Step 8.5: 计算光的偏振状态 ---
def classify_state(s3):
    if s3 is None:
        return "未知偏光"
    if -1.1 < s3 <= -0.9:
        return "左旋圆偏光"
    elif -0.9 < s3 <= -0.1:
        return "左旋椭圆偏光"
    elif -0.1 < s3 <= 0.1:
        return "线偏光"
    elif 0.1 < s3 <= 0.9:
        return "右旋椭圆偏光"
    elif 0.9 < s3 < 1.1:
        return "右旋圆偏光"
    else:
        return "未知偏光"


def compute_angle(s1, s2):
    if s1 is None or s2 is None:
        return ""
    angle = np.arctan2(s2, s1) * 90 / np.pi
    if angle < 0:
        angle += 180
    return f"{round(angle)}°"


s_data["State"] = s_data.apply(
    lambda row: compute_angle(row["S1*"], row["S2*"]) + classify_state(row["S3*"]),
    axis=1
)

# --- Step 9: 保存和输出结果 ---
output_path = os.path.join(os.path.dirname(file_paths[0]), "grayscale_stokes_output.txt")
s_data.to_csv(output_path, sep="	", index=False)
print(f"✅ Results saved to: {output_path}")
print(s_data.to_string(index=False))

# --- Step 10: 绘制球面图 (S1*, S2*, S3*) 为三维坐标 ---
from mpl_toolkits.mplot3d import Axes3D

# 方法一
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Stokes Sphere Visualization (S1*, S2*, S3*)")
# # ax.set_xlabel("S1*", labelpad=10)
# # ax.set_ylabel("S2*", labelpad=10)
# # ax.set_zlabel("S3*", labelpad=10)
#
# # 隐藏坐标轴刻度和边框面板
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# ax.zaxis.set_visible(False)
#
# # 彻底隐藏立方体边框面板
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor('white')
# ax.yaxis.pane.set_edgecolor('white')
# ax.zaxis.pane.set_edgecolor('white')
# ax.grid(False)
# ax.set_box_aspect([1, 1, 1])  # 保持轴比例一致
# ax.set_facecolor('white')    # 设置背景为纯白
# # 隐藏立方体边框（最底层框线）
# ax.set_frame_on(False)  # 彻底隐藏坐标框线
# fig.patch.set_facecolor('white')
#
# # 绘制球面
# u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
# x = np.cos(u) * np.sin(v)
# y = np.sin(u) * np.sin(v)
# z = np.cos(v)
# ax.plot_surface(x, y, z, color='lightblue', alpha=0.2, edgecolor='none')
#
# # 绘制坐标轴从原点出发
# ax.quiver(0, 0, 0, 1.8, 0, 0, color='black', arrow_length_ratio=0.05)
# ax.quiver(0, 0, 0, 0, -1.8, 0, color='black', arrow_length_ratio=0.05)  # 反向 S1*
# ax.quiver(0, 0, 0, 0, 0, 1.8, color='black', arrow_length_ratio=0.05)
#
# # 标注坐标轴标签在箭头旁
# ax.text(1.9, 0, 0, 'S2*', color='black')
# ax.text(0, -2.2, 0, 'S1*', color='black')
# ax.text(0, 0, 1.8, 'S3*', color='black')
#
# # 画球壳参考（使用透明球体，不显示网格）
# # u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
# # x = np.cos(u) * np.sin(v)
# # y = np.sin(u) * np.sin(v)
# # z = np.cos(v)
# # ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.3, alpha=0.5)
#
# # 绘制每个点
# for _, row in s_data.iterrows():
#     if pd.notna(row["S1*"]) and pd.notna(row["S2*"]) and pd.notna(row["S3*"]):
#         ax.scatter(row["S2*"], row["S1*"], row["S3*"], color='red')
#         ax.text(row["S2*"], row["S1*"], row["S3*"], str(int(row["Index"])), color='blue')
#
# plt.show()

# # 方法二
# import plotly.graph_objects as go
#
# # 生成球面数据
# u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
# x = np.cos(u) * np.sin(v)
# y = np.sin(u) * np.sin(v)
# z = np.cos(v)
#
# # 创建球面图层
# sphere = go.Surface(
#     x=x, y=y, z=z,
#     colorscale=[[0, 'lightblue'], [1, 'lightblue']],
#     opacity=0.2,
#     showscale=False
# )
#
# # 数据点及编号
# points = go.Scatter3d(
#     x=s_data["S1*"],
#     y=s_data["S2*"],
#     z=s_data["S3*"],
#     mode='markers+text',
#     marker=dict(size=5, color='red'),
#     text=[str(int(i)) for i in s_data["Index"]],
#     textposition="top center",
#     textfont=dict(color='blue')
# )
#
# # 坐标轴箭头
# axes = [
#     go.Scatter3d(x=[0, 1.2], y=[0, 0], z=[0, 0], mode='lines+text', line=dict(color='black'), text=['', 'S1*'], textposition='top right'),
#     go.Scatter3d(x=[0, 0], y=[0, 1.2], z=[0, 0], mode='lines+text', line=dict(color='black'), text=['', 'S2*'], textposition='top right'),
#     go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.2], mode='lines+text', line=dict(color='black'), text=['', 'S3*'], textposition='top right')
# ]
#
# # 绘制图像
# fig = go.Figure(data=[sphere, points] + axes)
# fig.update_layout(
#     title="Stokes Sphere Visualization (S1*, S2*, S3*)",
#     scene=dict(
#         xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
#         yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
#         zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
#         aspectmode='data',
#         bgcolor='white'
#     ),
#     showlegend=False
# )
# fig.show()

# --- Step 10: 使用 PyVista 绘制球面图 (S1*, S2*, S3*) 为三维坐标 ---
import pyvista as pv
from pyvista import themes

# 创建一个球体
sphere = pv.Sphere(radius=1.0, theta_resolution=120, phi_resolution=120)

# 创建 Plotter
plotter = pv.Plotter()
plotter.set_background("white")

# 添加球体，设置透明度和颜色
plotter.add_mesh(sphere, color="blue", opacity=0.2, show_edges=False)

# 添加坐标轴箭头（S1*, S2*, S3*）
# 使用 pv.Arrow 创建细坐标轴箭头
arrow_x = pv.Arrow(start=(0, 0, 0), direction=(1.2, 0, 0), tip_length=0.05, tip_radius=0.02, shaft_radius=0.01)
arrow_y = pv.Arrow(start=(0, 0, 0), direction=(0, -1.2, 0), tip_length=0.05, tip_radius=0.02, shaft_radius=0.01)
arrow_z = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1.2), tip_length=0.05, tip_radius=0.02, shaft_radius=0.01)

plotter.add_mesh(arrow_x, color="black")
plotter.add_mesh(arrow_y, color="black")
plotter.add_mesh(arrow_z, color="black")
# 添加坐标轴负向的细直线
plotter.add_lines(np.array([[0, 0, 0], [-1.2, 0, 0]]), color="black", width=2)
plotter.add_lines(np.array([[0, 0, 0], [0, 1.2, 0]]), color="black", width=2)
plotter.add_lines(np.array([[0, 0, 0], [0, 0, -1.2]]), color="black", width=2)

# 添加轴标签
plotter.add_point_labels(np.array([[1.2, 0, 0], [0, -1.2, 0], [0, 0, 1.2]]), ["S2*", "S1*", "S3*"], text_color="black")

# 添加数据点
points = np.array(list(zip(s_data["S2*"], s_data["S1*"], s_data["S3*"])) )
plotter.add_points(points, color='red', point_size=15, render_points_as_spheres=True)

# 计算偏移标签坐标（从球心向外拉远一点）
offset_labels = []
for pt in points:
    direction = pt / np.linalg.norm(pt) if np.linalg.norm(pt) != 0 else np.array([0, 0, 0])
    offset_point = pt + 0.05 * direction  # 偏移比例可调整
    offset_labels.append(offset_point)

offset_labels = np.array(offset_labels)

# 添加偏移后的标签
labels = [str(int(i)) for i in s_data["Index"]]
plotter.add_point_labels(
    offset_labels,
    labels,
    font_size=10,
    text_color="black",
    point_size=0.1,
    always_visible=True
)



# 启动交互窗口
plotter.show(title="Stokes Sphere Visualization", window_size=[800, 800])

