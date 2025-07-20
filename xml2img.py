import os
import xml.etree.ElementTree as ET
import numpy as np
from skimage.io import imread, imsave
from skimage.draw import polygon


def create_folder(path):
    """Ensure the directory exists"""
    os.makedirs(path, exist_ok=True)


def xml_to_point_png(label_xml_dir, img_dir, label_point_dir, img_list=None):
    """
    Convert XML region annotations into pixel-level point masks (PNG).

    Parameters:
    - label_xml_dir: directory containing .xml annotation files
    - img_dir: directory containing original images (to read image size)
    - label_point_dir: output directory for _label_point.png masks
    - img_list: list of image filenames to process (with extensions). If None, lists all .xml in label_xml_dir.
    """
    create_folder(label_point_dir)

    # If no explicit image list is provided, derive from XML filenames
    if img_list is None:
        img_list = [f.split('.xml')[0] + ext for f in os.listdir(label_xml_dir) 
                    if f.lower().endswith('.xml')
                    for ext in ['.png']]

    for image_name in sorted(img_list):
        name, _ = os.path.splitext(image_name)
        xml_path = os.path.join(label_xml_dir, f"{name}.xml")
        img_path = os.path.join(img_dir, image_name)

        # Read image to get dimensions
        img = imread(img_path)
        h, w = img.shape[:2]

        # Initialize empty mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Parse XML and fill each Region polygon
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for region in root.findall('.//Region'):
            xs, ys = [], []
            for vertex in region.findall('.//Vertex'):
                xs.append(float(vertex.attrib['X']))
                ys.append(float(vertex.attrib['Y']))
            if xs and ys:
                rr, cc = polygon(ys, xs, (h, w))
                mask[rr, cc] = 255  # white fill

        # Save the point mask PNG
        out_path = os.path.join(label_point_dir, f"{name}_label_point.png")
        imsave(out_path, mask)
        print(f"Saved point mask: {out_path}")


if __name__ == '__main__':
    # --- User Configuration ---
    label_xml_dir   = 'data/MO/xml'      # e.g., '../data/MO/xml'
    img_dir         = 'data/MO/images'    # e.g., '../data/MO/images'
    label_point_dir = 'data/MO/label_point'   # e.g., '../data/MO/label_point'

    # Optionally specify images to process or let the script infer from XML
    img_list = None  # or ['img1.png', 'img2.png', ...]

    xml_to_point_png(label_xml_dir, img_dir, label_point_dir, img_list)
