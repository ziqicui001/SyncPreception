#pip install collections

import os
import cv2
import numpy as np
import pandas as pd
from collections import namedtuple
from shapely.geometry import Polygon

# Define the label structure
Label = namedtuple('Label', ['label_name', 'class_id', 'colorRGB'])

def extract_color_polygons(image, fixed_color, tolerance=2):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_bound = np.array(fixed_color) - tolerance
    upper_bound = np.array(fixed_color) + tolerance
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Optional: Apply some morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check the OpenCV version
    if int(cv2.__version__.split('.')[0]) < 4:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    polygons = [contour.reshape(-1, 2).tolist() for contour in contours]
    return polygons

def calculate_centroid(polygon):
    if len(polygon) < 3:
        return None, None
    shapely_polygon = Polygon(polygon)
    if not shapely_polygon.is_valid:
        return None, None
    centroid = shapely_polygon.centroid
    return int(centroid.x), int(centroid.y)

def calculate_polygon_area(polygon):
    if len(polygon) < 3:
        return 0
    shapely_polygon = Polygon(polygon)
    if not shapely_polygon.is_valid:
        return 0
    return shapely_polygon.area

def process_single_image(image_path, output_path, labels_ade):
    """
    Processes a single image, extracting color polygons based on the labels provided.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path where the output CSV file will be saved.
    - labels_ade: List of Label namedtuples containing label_name, class_id, and colorRGB.
    """
    image = cv2.imread(image_path)
    image_shape = image.shape
    image_area = image_shape[0] * image_shape[1]
    data = []
    count = 0  # Start Node_id from 0
    for label in labels_ade:
        polygons = extract_color_polygons(image, label.colorRGB)
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            centroid_x, centroid_y = calculate_centroid(polygon)
            if centroid_x is None or centroid_y is None:
                continue
            polygon_area = calculate_polygon_area(polygon)
            percentage_area = (polygon_area / image_area) * 100
            data.append({
                'Node_id': count,
                'name': label.label_name,
                'Item_id': label.class_id,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'proportion': percentage_area,
                'polygon': polygon
            })
            count += 1
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = os.path.splitext(os.path.basename(image_path))[0] + '.csv'
    csv_path = os.path.join(output_path, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Processed {image_path} and saved results to {csv_path}")

def run_color_extraction_for_image(image_path, output_dir, labels):
    """
    Main function to process a single image for color extraction and save results.

    Parameters:
    - image_path: Path to the input image file.
    - output_dir: Path to the directory where output CSV file will be saved.
    - labels: List of Label namedtuples containing label_name, class_id, and colorRGB.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_single_image(image_path, output_dir, labels)

# Example usage
if __name__ == '__main__':
    # Define the ADE20K labels
    labels_ade = [
        Label('wall', 0, (120, 120, 120)),
        Label('building', 1, (180, 120, 120)),
        Label('sky', 2, (6, 230, 230)),
        Label('floor', 3, (80, 50, 50)),
        Label('tree', 4, (4, 200, 3)),
        Label('ceiling', 5, (120, 120, 80)),
        Label('road', 6, (140, 140, 140)),
        Label('grass', 9, (4, 250, 7)),
        Label('sidewalk', 11, (235, 255, 7)),
        Label('person', 12, (150, 5, 61)),
        Label('earth', 13, (120, 120, 70)),
        Label('door', 14, (8, 255, 51)),
        Label('table', 15, (255, 6, 82)),
        Label('mountain', 16, (143, 255, 140)),
        Label('plant', 17, (204, 255, 4)),
        Label('chair', 19, (204, 70, 3)),
        Label('car', 20, (0, 102, 200)),
        Label('water', 21, (61, 230, 250)),
        Label('house', 25, (255, 9, 224)),
        Label('sea', 26, (9, 7, 230)),
        Label('field', 29, (112, 9, 255)),
        Label('armchair', 30, (8, 255, 214)),
        Label('seat', 31, (7, 255, 224)),
        Label('fence', 32, (255, 184, 6)),
        Label('desk', 33, (10, 255, 71)),
        Label('rock', 34, (255, 41, 10)),
        Label('railing', 38, (255, 61, 6)),
        Label('base', 40, (255, 122, 8)),
        Label('column', 42, (255, 8, 41)),
        Label('signboard', 43, (255, 5, 153)),
        Label('sand', 46, (160, 150, 20)),
        Label('grandstand', 51, (31, 255, 0)),
        Label('path', 52, (255, 31, 0)),
        Label('stairs', 53, (255, 224, 0)),
        Label('runway', 54, (153, 255, 0)),
        Label('screen door', 58, (0, 173, 255)),
        Label('stairway', 59, (31, 0, 255)),
        Label('river', 60, (11, 200, 200)),
        Label('bridge', 61, (255, 82, 0)),
        Label('coffee table', 64, (0, 255, 112)),
        Label('flower', 66, (255, 0, 0)),
        Label('hill', 68, (255, 102, 0)),
        Label('bench', 69, (194, 255, 0)),
        Label('palm', 72, (0, 82, 255)),
        Label('swivel chair', 75, (10, 0, 255)),
        Label('boat', 76, (173, 255, 0)),
        Label('bar', 77, (0, 255, 153)),
        Label('hovel', 79, (255, 0, 255)),
        Label('bus', 80, (255, 0, 245)),
        Label('light', 82, (255, 173, 0)),
        Label('truck', 83, (255, 0, 20)),
        Label('tower', 84, (255, 184, 184)),
        Label('awning', 86, (0, 255, 61)),
        Label('streetlight', 87, (0, 71, 255)),
        Label('booth', 88, (255, 0, 204)),
        Label('airplane', 90, (0, 255, 82)),
        Label('dirt track', 91, (0, 10, 255)),
        Label('pole', 93, (51, 0, 255)),
        Label('land', 94, (0, 194, 255)),
        Label('bannister', 95, (0, 122, 255)),
        Label('stage', 101, (82, 0, 255)),
        Label('van', 102, (163, 255, 0)),
        Label('ship', 103, (255, 235, 0)),
        Label('fountain', 104, (8, 184, 170)),
        Label('canopy', 106, (0, 255, 92)),
        Label('swimming pool', 109, (0, 184, 255)),
        Label('stool', 110, (0, 214, 255)),
        Label('barrel', 111, (255, 0, 112)),
        Label('waterfall', 113, (0, 224, 255)),
        Label('tent', 114, (112, 224, 255)),
        Label('minibike', 116, (163, 0, 255)),
        Label('step', 121, (255, 0, 143)),
        Label('pot', 125, (245, 0, 255)),
        Label('animal', 126, (255, 0, 122)),
        Label('bicycle', 127, (255, 245, 0)),
        Label('lake', 128, (10, 190, 212)),
        Label('screen', 130, (0, 204, 255)),
        Label('sculpture', 132, (255, 255, 0)),
        Label('traffic light', 136, (41, 0, 255)),
        Label('ashcan', 138, (173, 0, 255)),
        Label('pier', 140, (71, 0, 255)),
        Label('crt screen', 141, (122, 0, 255)),
        Label('plate', 142, (0, 255, 184)),
        Label('bulletin board', 144, (184, 255, 0))
    ]

    # Define input image path and output directory
    image_path = 't01.2.png'  #原始街景图
    output_dir = 'nouse'     #导出最基础的node csv

    # Run the color extraction process for a single image
    run_color_extraction_for_image(image_path, output_dir, labels_ade)
