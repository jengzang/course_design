import matplotlib

matplotlib.use('TkAgg')  # 强制使用 GUI 窗口

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from googletrans import Translator
import os


# 翻译中文为英文
def translate_text(text):
    try:
        return Translator().translate(text, src='zh-cn', dest='en').text
    except:
        return text


selected_area = []
selection_done = False


def on_select(eclick, erelease):
    global selection_done
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    selected_area.clear()
    selected_area.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    selection_done = True
    print(f"Selected region: ({x1}, {y1}) to ({x2}, {y2})")


# 打开文件选择器，支持多图
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select image files",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
)

if file_paths:
    # 输出路径设定（与图像同一文件夹）
    output_folder = os.path.dirname(file_paths[0])
    output_txt_path = os.path.join(output_folder, "grayscale_results.txt")
    results = []

    # 显示第一张图供用户框选
    first_image = Image.open(file_paths[0]).convert("L")
    fig, ax = plt.subplots()
    ax.imshow(first_image, cmap='gray')
    ax.set_title(translate_text("拖动鼠标选择区域，松开后应用到所有图像"))

    selector = RectangleSelector(ax, on_select, useblit=True, button=[1],
                                 minspanx=5, minspany=5, spancoords='pixels',
                                 interactive=True)

    print("🖱 Please drag to select region on the first image...")
    while not selection_done:
        plt.pause(0.1)
    plt.close()

    # 应用选区到所有图像
    print("\n📊 Grayscale Averages:")
    for path in file_paths:
        try:
            image = Image.open(path).convert("L")
            x1, y1, x2, y2 = selected_area[0]

            width, height = image.size
            x1_c, x2_c = max(0, x1), min(width, x2)
            y1_c, y2_c = max(0, y1), min(height, y2)

            pixels = [
                image.getpixel((x, y))
                for y in range(y1_c, y2_c)
                for x in range(x1_c, x2_c)
            ]
            avg = sum(pixels) / len(pixels) if pixels else 0
            file_name = os.path.splitext(os.path.basename(path))[0]
            result_line = f"{file_name}\t{avg:.2f}"

            print(result_line)
            results.append(result_line)

            # ✅ 保存灰度图像
            gray_save_path = os.path.join(
                os.path.dirname(path),
                f"{file_name}_gray{os.path.splitext(path)[1]}"
            )
            image.save(gray_save_path)

        except Exception as e:
            print(f"{path} → Error: {e}")

    # 写入 txt 文件
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\n✅ Results saved to: {output_txt_path}")
else:
    print("No images selected.")
