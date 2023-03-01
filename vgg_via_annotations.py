import os
from pathlib import Path

import pandas as pd
import skimage
from tqdm.auto import tqdm

pd.set_option('display.max_columns', 15)

from src.config import DATA_PATH


def poly_coords(obj, mask_height, mask_width):
    import numpy as np
    coordinates = []
    # create black image an attach the object patch
    obj_mask = np.zeros([mask_height, mask_width], dtype=np.uint8)
    obj_mask[obj._slice] = obj.image

    # extend coordinates with all the contours segments of the object
    contours = skimage.measure.find_contours(
        obj_mask, fully_connected='high', positive_orientation='high')
    for contour in contours:
        coordinates.extend([(int(p[1]), int(p[0])) for p in contour])

    # downsample contour coordinates to 40 points
    idx = np.round(np.linspace(0, len(coordinates) - 1, 10)).astype(int)
    coordinates = np.array(coordinates).round(2)
    coordinates = coordinates[idx].tolist()
    return coordinates


def contour2viapoly(contour: list):
    """
    Transform object's contour coordinates in the mask to VGG VIA polygon format for the csv annotation.

    :param contour: list of points coordinates (i.e.: [[x1, y1], [x2, y2], ...], where x*, y* are expressed as pixel
    in the mask
    :return:
    """
    all_x, all_y = list(), list()
    for point in contour:
        all_x.append(point[0])
        all_y.append(point[1])

    return all_x, all_y


def get_annotations(iterator, iterations):
    # initialize annotations df
    annotations_df = pd.DataFrame({}, columns=['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
                                              'region_shape_attributes', 'region_attributes'])
    cols = annotations_df.columns

    for item in tqdm(iterator, total=iterations, leave=True, desc="image loop"):
        if len(item) == 2:
            img_name, mask_name = item
        else:
            img_name, mask_name = item, item
        mask_orig = skimage.io.imread(MASKS_PATH / mask_name, as_gray=True)
        img_path = MASKS_PATH.parent.parent / 'images'
        img_size = os.stat(img_path / img_name).st_size

        label, n_objs = skimage.measure.label(mask_orig, return_num=True)

        for id_obj, obj in tqdm(enumerate(skimage.measure.regionprops(label)), leave=False, desc="objects loop"):
            annotations = poly_coords(obj, mask_orig.shape[0], mask_orig.shape[1])
            all_x, all_y = contour2viapoly(annotations)

            # initialize record
            record = pd.DataFrame([[img_name, img_size, {}, n_objs, id_obj, {
                "name": "polygon",
                "all_points_x": all_x,
                "all_points_y": all_y,
            },
                                    {}]], columns=cols)
            annotations_df = pd.concat([annotations_df, record])
    return annotations_df


def format_annotations(annotations_fn: Path):
    # read annotations df as text file
    with open(annotations_fn, 'r') as f:
        text = f.read()

    # VGG VIA compatible formatting
    formatted_text = text.replace("'", "\"\"")

    # save formatted annotations
    with open(annotations_fn.parent / f"{annotations_fn.name.replace('_csv', '_formatted_csv')}", 'w') as f:
        f.write(formatted_text)
    return


if __name__ == '__main__':
    th = 0.55
    MASKS_PATH = DATA_PATH / 'masks' / f'{th}'
    mask_paths_list = [*MASKS_PATH.iterdir()]

    iterator = [p.name for p in mask_paths_list]
    iterations = len(iterator)

    annotations_df = get_annotations(iterator, iterations)
    outpath = MASKS_PATH.parent.parent
    annotations_df.to_csv(outpath / 'pre_annotations_csv.csv', index=False)
    format_annotations(outpath / 'pre_annotations_csv.csv')
