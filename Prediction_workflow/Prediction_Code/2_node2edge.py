import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import ast
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from itertools import combinations
import os
from matplotlib.collections import PatchCollection

# 转换多边形数据格式
def convert_polygon_format(polygon_str):
    try:
        points_list = ast.literal_eval(polygon_str)
        return [(float(x), float(y)) for x, y in points_list]
    except Exception as e:
        print(f"Error parsing polygon: {polygon_str} with error {e}")
        return None

def process_single_file(image_path, csv_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    image = cv2.imread(image_path)
    df = pd.read_csv(csv_path, encoding="latin")

    image_height, image_width, _ = image.shape

    df['polygon'] = df['polygon'].apply(convert_polygon_format)
    df = df[df['polygon'].notnull()]
    df['polygon'] = df['polygon'].apply(lambda x: Polygon(x) if x else None)
    df = df[df['polygon'].notnull()]

    connected_pairs = []

    kernel_size = 2
    image_bounds = box(0, 0, image_width, image_height)

    df['buffered_polygon'] = df['polygon'].apply(lambda x: x.buffer(kernel_size).intersection(image_bounds))

    for (idx1, row1), (idx2, row2) in combinations(df.iterrows(), 2):
        poly1 = row1['buffered_polygon']
        poly2 = row2['buffered_polygon']

        if poly1.intersects(poly2) or poly1.touches(poly2):
            print(f"Intersection or touch found between {row1['Node_id']} and {row2['Node_id']}")
            connected_pairs.append((row1['Node_id'], row2['Node_id'], row1['centroid_x'], row1['centroid_y'], row2['centroid_x'], row2['centroid_y']))

    connected_df = pd.DataFrame(connected_pairs, columns=['ID1', 'ID2', 'centroid_x1', 'centroid_y1', 'centroid_x2', 'centroid_y2'])

    connected_df['length'] = connected_df.apply(lambda row: np.sqrt((row['centroid_x2'] - row['centroid_x1'])**2 + (row['centroid_y2'] - row['centroid_y1'])**2), axis=1)
    connected_df = connected_df[['ID1', 'ID2', 'length']]

    output_filename = f"{os.path.splitext(os.path.basename(csv_path))[0]}_edgelist.csv"
    output_path = os.path.join(output_folder_path, output_filename)

    connected_df.to_csv(output_path, index=False)

    print(f"新CSV文件已保存到 {output_path}")
    print(connected_df)

# 示例使用
image_path = 't01.2.png'  # 输入原始街景图
csv_path = 't01.2.csv'  # 输入最基础的node.csv
output_folder_path = 'nouse'  # 输出保存edgelist.csv

process_single_file(image_path, csv_path, output_folder_path)