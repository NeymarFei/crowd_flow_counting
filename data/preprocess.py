import json
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(__file__)
INPUT_ROOT = os.path.join(BASE_DIR, "labels")
OUTPUT_ROOT = os.path.join(BASE_DIR, "scene_reports")

# 确保输出目录首先被创建
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def parse_labelme_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_w = data.get('imageWidth', 720)
    img_h = data.get('imageHeight', 576)
    pts = [shape['points'][0] for shape in data['shapes'] if shape['label'] in ['pedestrian', 'outflow']]
    return len(pts), pts, img_w, img_h


def main():
    # 自查：看看 labels 文件夹是否存在
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 错误：找不到 labels 文件夹，请确保它在: {INPUT_ROOT}")
        return

    # 1. 获取所有子文件夹
    scenes = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]

    if not scenes:
        print(f"❓ 警告：在 {INPUT_ROOT} 中没有发现子文件夹！")
        print(f"当前 labels 目录下的内容有: {os.listdir(INPUT_ROOT)}")
        return

    print(f"✅ 检测到 {len(scenes)} 个场景文件夹，准备分析...")
    summary_list = []

    for scene_name in tqdm(scenes, desc="总进度"):
        scene_path = os.path.join(INPUT_ROOT, scene_name)
        json_files = sorted(glob.glob(os.path.join(scene_path, "*.json")), key=natural_sort_key)

        if not json_files:
            print(f"跳过 {scene_name}: 文件夹内没有 JSON 文件")
            continue

        scene_output_dir = os.path.join(OUTPUT_ROOT, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        counts = []
        all_points = []
        w, h = 720, 576  # 默认值

        for file in json_files:
            try:
                c, pts, w, h = parse_labelme_json(file)
                counts.append(c)
                all_points.extend(pts)
            except Exception as e:
                print(f"文件 {file} 解析出错: {e}")

        # 绘图逻辑
        if counts:
            # 1. 波动图
            plt.figure(figsize=(10, 4))
            plt.plot(range(len(counts)), counts, color='#1f77b4', linewidth=2)
            plt.title(f"Scene: {scene_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(scene_output_dir, "variation_plot.png"))
            plt.close()

            # 2. 热力图
            heatmap = np.zeros((h, w), dtype=np.float32)
            for x, y in all_points:
                if 0 <= int(x) < w and 0 <= int(y) < h:
                    heatmap[int(y), int(x)] += 1
            k_size = int(w / 15) | 1
            heatmap_blur = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
            if heatmap_blur.max() > 0:
                norm = (heatmap_blur / heatmap_blur.max() * 255).astype(np.uint8)
                color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(scene_output_dir, "spatial_heatmap.png"), color_map)

            summary_list.append({
                "场景编号": scene_name,
                "总帧数": len(counts),
                "平均人数": round(np.mean(counts), 2),
                "峰值人数": np.max(counts)
            })

    # 保存总表
    if summary_list:
        df_summary = pd.DataFrame(summary_list)
        df_summary.to_csv(os.path.join(OUTPUT_ROOT, "all_scenes_summary.csv"), index=False, encoding='utf_8_sig')
        print(f"\n✨ 分析完成！结果已保存在: {OUTPUT_ROOT}")
    else:
        print("\n数据为空，未生成汇总表。")


if __name__ == "__main__":
    main()
