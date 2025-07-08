import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from matplotlib.patches import Ellipse
from PIL import Image
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
active_listeners = []  # 保存所有激活的事件监听 ID
active_preview = []  # 保存当前预览对象（如线、椭圆）


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

# --- Step 3: 用户图形界面选择采样工具 ---
img = Image.open(file_paths[0]).convert("L")
ref_points = []
sampling_complete = False

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.imshow(img, cmap='gray')
ax.set_title("选择采样工具")

# 按钮区域
ax_line = plt.axes([0.1, 0.05, 0.2, 0.075])
ax_ellipse = plt.axes([0.4, 0.05, 0.2, 0.075])
ax_free = plt.axes([0.7, 0.05, 0.2, 0.075])

# 苹果灰风格按钮
btn_line = Button(ax_line, '直线工具', color='#dcdcdc', hovercolor='#a9a9a9')  # lightgray + darkgray
btn_ellipse = Button(ax_ellipse, '椭圆工具', color='#dcdcdc', hovercolor='#a9a9a9')
btn_free = Button(ax_free, '描点工具', color='#dcdcdc', hovercolor='#a9a9a9')


def redraw_result():
    fig2, ax2 = plt.subplots()
    ax2.imshow(img, cmap='gray')
    for i, (x, y) in enumerate(ref_points):
        ax2.plot(x, y, 'ro', markersize=3)
        ax2.text(x + 4, y + 4, str(i + 1), color='yellow', fontsize=7)
    xs, ys = zip(*ref_points)
    ax2.plot(xs, ys, 'r-', lw=1)
    ax2.set_title("采样结果预览")
    mng = plt.get_current_fig_manager()
    backend = plt.get_backend()
    if backend == "TkAgg":
        try:
            mng.resize(*mng.window.wm_maxsize())  # Tkinter 最大化
        except Exception as e:
            print("⚠️ TkAgg 无法最大化窗口：", e)

    elif backend.startswith("Qt"):
        try:
            mng.window.showMaximized()
        except Exception as e:
            print("⚠️ Qt 后端最大化失败：", e)

    elif backend == "WXAgg":
        try:
            mng.frame.Maximize(True)
        except Exception as e:
            print("⚠️ WX 后端最大化失败：", e)

    plt.show(block=False)
    plt.pause(0.1)


def clear_previous_tool():
    # 清除事件监听
    for cid in active_listeners:
        fig.canvas.mpl_disconnect(cid)
    active_listeners.clear()

    # 清除预览图形对象
    for artist in active_preview:
        try:
            artist.remove()
        except Exception:
            pass
    active_preview.clear()

    fig.canvas.draw_idle()


def line_tool(event):
    global ref_points, sampling_complete
    clear_previous_tool()
    ref_points.clear()
    ax.set_title("点击两点定义一条线（右键撤销）")
    fig.canvas.draw()

    preview_line, = ax.plot([], [], '--', color='red')
    active_preview.append(preview_line)

    points = []

    def on_move(evt):
        if len(points) == 1 and evt.inaxes == ax:
            x0, y0 = points[0]
            x1, y1 = evt.xdata, evt.ydata
            preview_line.set_data([x0, x1], [y0, y1])
            fig.canvas.draw_idle()

    def on_click(evt):
        if evt.inaxes != ax:
            return
        if evt.button == 3 and points:
            points.pop()  # 右键撤销
            preview_line.set_data([], [])
            fig.canvas.draw_idle()
            return

        points.append((evt.xdata, evt.ydata))
        if len(points) == 2:
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_move)
            preview_line.remove()

            (x0, y0), (x1, y1) = points
            length = int(np.hypot(x1 - x0, y1 - y0))
            n = length // 10
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            ref_points.extend(zip(xs, ys))
            ax.plot(xs, ys, 'r-', lw=1)
            for i, (x, y) in enumerate(ref_points):
                ax.plot(x, y, 'ro', markersize=3)
                ax.text(x + 4, y + 4, str(i + 1), color='yellow', fontsize=7)
            fig.canvas.draw()

            # sampling_complete = True
            print("✅ 采样完成")
            for i, pt in enumerate(ref_points):
                print(f"Point {i + 1}: ({int(pt[0])}, {int(pt[1])})")
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    active_listeners.append(cid_click)

    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)
    active_listeners.append(cid_move)


def ellipse_tool(event):
    global ref_points, sampling_complete
    clear_previous_tool()
    ref_points.clear()
    ax.set_title("点击两点定义椭圆（右键撤销）")
    fig.canvas.draw()

    from matplotlib.patches import Ellipse
    preview_ellipse = Ellipse((0, 0), 0, 0, fill=False, color='red', linestyle='--')
    ax.add_patch(preview_ellipse)
    active_preview.append(preview_ellipse)

    points = []

    def on_move(evt):
        if len(points) == 1 and evt.inaxes == ax:
            x0, y0 = points[0]
            x1, y1 = evt.xdata, evt.ydata
            xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
            a, b = abs(x1 - x0), abs(y1 - y0)
            preview_ellipse.set_center((xc, yc))
            preview_ellipse.width = a
            preview_ellipse.height = b
            fig.canvas.draw_idle()

    def on_click(evt):
        if evt.inaxes != ax:
            return
        if evt.button == 3 and points:
            points.pop()
            preview_ellipse.set_width(0)
            preview_ellipse.set_height(0)
            fig.canvas.draw_idle()
            return

        points.append((evt.xdata, evt.ydata))
        if len(points) == 2:
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_move)
            preview_ellipse.remove()

            (x0, y0), (x1, y1) = points
            xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
            a, b = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            n = int(np.pi * (a + b)) // 10
            theta = np.linspace(0, 2 * np.pi, n)
            xs = xc + a * np.cos(theta)
            ys = yc + b * np.sin(theta)
            ref_points.extend(zip(xs, ys))
            e = Ellipse((xc, yc), width=2 * a, height=2 * b, fill=False, color='cyan')
            ax.add_patch(e)
            ax.plot(xs, ys, 'r-', lw=1)
            for i, (x, y) in enumerate(ref_points):
                ax.plot(x, y, 'ro', markersize=3)
                ax.text(x + 4, y + 4, str(i + 1), color='yellow', fontsize=7)
            fig.canvas.draw()

            sampling_complete = True
            print("✅ 采样完成")
            for i, pt in enumerate(ref_points):
                print(f"Point {i + 1}: ({int(pt[0])}, {int(pt[1])})")
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    active_listeners.append(cid_click)

    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)
    active_listeners.append(cid_move)


def free_tool(event):
    global ref_points, sampling_complete
    clear_previous_tool()
    ref_points.clear()
    ax.set_title("点击绘制路径（右键撤销，回车结束）")
    fig.canvas.draw()

    coords = []
    preview_line, = ax.plot([], [], '--', color='red')
    active_preview.append(preview_line)

    def on_move(evt):
        if len(coords) >= 1 and evt.inaxes == ax:
            xs, ys = zip(*coords)
            xs += (evt.xdata,)
            ys += (evt.ydata,)
            preview_line.set_data(xs, ys)
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'enter':
            done_sampling()

    def on_click(evt):
        if evt.inaxes != ax:
            return
        if evt.button == 3 and coords:  # 右键撤销
            coords.pop()
            preview_line.set_data(zip(*coords) if coords else ([], []))
            fig.canvas.draw_idle()
        else:
            coords.append((evt.xdata, evt.ydata))
            ax.plot(evt.xdata, evt.ydata, marker='o', color='blue', markersize=3)
            fig.canvas.draw_idle()

    def done_sampling():
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_move)
        fig.canvas.mpl_disconnect(cid_key)
        preview_line.remove()

        if len(coords) < 2:
            return
        sampled = [coords[0]]
        for i in range(1, len(coords)):
            x0, y0 = coords[i - 1]
            x1, y1 = coords[i]
            d = int(np.hypot(x1 - x0, y1 - y0))
            n = max(2, d // 10)
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            sampled.extend(list(zip(xs[1:], ys[1:])))
        ref_points.extend(sampled)
        xs, ys = zip(*ref_points)
        ax.plot(xs, ys, 'r-', lw=1)
        for i, (x, y) in enumerate(ref_points):
            ax.plot(x, y, 'ro', markersize=3)
            ax.text(x + 4, y + 4, str(i + 1), color='yellow', fontsize=7)
        fig.canvas.draw()

        sampling_complete = True
        print("✅ 采样完成")
        for i, pt in enumerate(ref_points):
            print(f"Point {i + 1}: ({int(pt[0])}, {int(pt[1])})")
        plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    active_listeners.append(cid_click)

    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)
    active_listeners.append(cid_move)

    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)


btn_line.on_clicked(line_tool)
btn_ellipse.on_clicked(ellipse_tool)
btn_free.on_clicked(free_tool)

plt.show()

# if not sampling_complete:
#     exit("❌ 未完成采样")
#
# print("✅ 采样完成：")
# for i, pt in enumerate(ref_points):
#     print(f"Point {i + 1}: ({int(pt[0])}, {int(pt[1])})")

# --- Step 3.5: 转换为相对坐标 ---
ref_origin = origin_points[0]
relative_coords = [(int(x - ref_origin[0]), int(y - ref_origin[1])) for (x, y) in ref_points]
point_labels = [f"p{i + 1}" for i in range(len(relative_coords))]

# --- Step 4: 初始化结果表 ---
results = []
gray_labels = ["9045", "090", "045", "00"]
suffix_to_gray = {"00": "gray(0,0)", "090": "gray(0,90)", "045": "gray(0,45)", "9045": "gray(90,45)"}
gray_table = {label: {} for label in point_labels}

# --- Step 5: 提取每个图像中每个点的灰度值 ---
for path, origin in zip(file_paths, origin_points):
    img = Image.open(path).convert("L")
    filename = os.path.splitext(os.path.basename(path))[0]
    max_i = max_i_dict.get(filename, 1.0)
    print(f"✅ 获取光强 {max_i} 用于图像 {filename}")

    gray_name = None
    for suffix in gray_labels:
        if filename.endswith(suffix):
            gray_name = suffix_to_gray[suffix]
            break
    if gray_name is None:
        print(f"⚠️ 文件 {filename} 不符合后缀（9045, 090, 045, 00）")
        continue

    for i, label in enumerate(point_labels):
        dx, dy = relative_coords[i]
        x, y = origin[0] + dx, origin[1] + dy
        gray_val = img.getpixel((x, y)) if 0 <= x < img.width and 0 <= y < img.height else None
        if gray_val is not None:
            gray_val *= max_i
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
    if pd.isna(s1) or pd.isna(s2):
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

redraw_result()

# --- Step 10: 使用 PyVista 绘图（包含连线 + 方向箭头） ---
import pyvista as pv
import numpy as np

plotter = pv.Plotter(window_size=[800, 800])
plotter.set_background("white")


# 添加球体
sphere = pv.Sphere(radius=1.0, theta_resolution=100, phi_resolution=50)
plotter.add_mesh(sphere, color="lightblue", opacity=0.2)
# 构造完整一圈赤道圆（单位圆，Z=0 平面）
theta = np.linspace(0, 2 * np.pi, 200)
x = np.cos(theta)
y = np.sin(theta)
z = np.zeros_like(theta)
circle_points = np.column_stack((x, y, z))

# 构造线段连接所有点首尾相连
lines = np.full((len(circle_points), 3), 2)
lines[:, 1] = np.arange(len(circle_points))
lines[:-1, 2] = np.arange(1, len(circle_points))
lines[-1, 2] = 0  # 最后一个点连回起点

circle = pv.PolyData()
circle.points = circle_points
circle.lines = lines

plotter.add_mesh(circle, color="gray", line_width=2)



# 添加正向箭头坐标轴
arrow_x = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=1.3,
                   tip_length=0.05, tip_radius=0.02, shaft_radius=0.01)

arrow_y = pv.Arrow(start=(0, 0, 0), direction=(0, -1, 0), scale=1.3,
                   tip_length=0.05, tip_radius=0.02, shaft_radius=0.01)

arrow_z = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=1.3,
                   tip_length=0.05, tip_radius=0.02, shaft_radius=0.01)
plotter.add_mesh(arrow_x, color="black")
plotter.add_mesh(arrow_y, color="black")
plotter.add_mesh(arrow_z, color="black")

# 添加负向坐标线
plotter.add_lines(np.array([[0, 0, 0], [-1.3, 0, 0]]), color="black", width=1.5)
plotter.add_lines(np.array([[0, 0, 0], [0, 1.3, 0]]), color="black", width=1.5)
plotter.add_lines(np.array([[0, 0, 0], [0, 0, -1.3]]), color="black", width=1.5)

# 添加轴标签
plotter.add_point_labels(np.array([[1.5, 0, 0], [0, -1.5, 0], [0, 0, 1.5]]),
                         ["S2*", "S1*", "S3*"], text_color="black")

# 添加数据点
points = np.array(list(zip(s_data["S2*"], s_data["S1*"], s_data["S3*"])))
plotter.add_points(points, color='red', point_size=10, render_points_as_spheres=True)

# 添加编号标签（稍微偏移 + 始终可见）
offset_labels = []
for pt in points:
    direction = pt / np.linalg.norm(pt) if np.linalg.norm(pt) != 0 else np.array([0, 0, 0])
    offset_labels.append(pt + 0.08 * direction)
labels = [str(int(i)) for i in s_data["Index"]]
plotter.add_point_labels(np.array(offset_labels), labels, font_size=15,
                         text_color="blue", always_visible=True)

# --- 添加每一段的方向箭头 ---
for i in range(len(points) - 1):
    p1 = points[i]
    p2 = points[i + 1]
    # 添加红线
    plotter.add_lines(np.array([p1, p2]), color="red", width=3)

    # 在中间加一个小箭头头部（没有直线部分）
    mid = (p1 + p2) / 2
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm == 0:
        continue
    unit_vec = direction / norm

    # 用 Cone 模拟箭头头部，起点 mid、方向 unit_vec
    arrow_head = pv.Cone(center=mid, direction=unit_vec,
                         height=0.03, radius=0.005, resolution=20)
    plotter.add_mesh(arrow_head, color="gray")

# 添加按钮控制的表格显示
table_shown = [False]  # 用列表包装以在闭包中可修改
table_actor = [None]


# 添加按钮控制的表格显示
table_shown = [False]
table_actor = [None]

def toggle_table(flag):
    if table_shown[0]:
        if table_actor[0]:
            plotter.remove_actor(table_actor[0])
            table_actor[0] = None
    else:
        # 格式化表格文本，使用等宽排列
        header = f"{'Index':<6} {'p':<6} {'State'}\n"
        # 显示前将 State 的中文翻译成英文（仅用于显示）
        state_map = {
            "左旋圆偏光": "L-Circ",
            "左旋椭圆偏光": "L-Ellip",
            "线偏光": "Linear",
            "右旋椭圆偏光": "R-Ellip",
            "右旋圆偏光": "R-Circ",
            "未知偏光": "Unknown"
        }

        def translate_state(full_state):
            for zh, en in state_map.items():
                if zh in full_state:
                    return full_state.replace(zh, en)
            return full_state  # fallback

        rows = [
            f"{i:<6} {p:<6} {translate_state(s)}"
            for i, p, s in zip(s_data["Index"], s_data["p"], s_data["State"])
        ]
        table_text = header + "\n".join(rows)
        table_actor[0] = plotter.add_text(table_text,
                                          position='upper_left',
                                          font_size=10,
                                          color='black',
                                          name='data_table')
    table_shown[0] = not table_shown[0]

# 添加左下角的按钮（避免遮挡表格）
plotter.add_checkbox_button_widget(toggle_table, value=False, position=(20, 20), size=25)

# 添加主标题
plotter.add_text("Poincaré sphere", position='upper_edge',
                 font_size=16, color='black')  # courier 比较粗


# 添加右下角脚注
footer_text = (
    "Innovative Design of Photoelectric Information\n"
    "Group 8: Yhx Cjt Yz Hyx Lym Dzh\n"
    "2025-7"
)

plotter.add_text(
    footer_text,
    position='lower_right',
    font_size=10,
    color='#333333',          # 深灰色
)


# 显示交互窗口
plotter.show(title="Stokes Sphere Visualization")
