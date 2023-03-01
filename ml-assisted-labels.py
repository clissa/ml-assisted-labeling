from src.image_loader import CellsLoader
from src.model.resunet import *
# from src.model.utils import *
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
from typing import List
from skimage import io
from pathlib import Path
from src.config import MODEL_PATH, DATA_PATH


def post_processing(preds: List[torch.Tensor], th: float, min_obj_size: int=12, foot: int=4, area_threshold: int=6, max_dist: int=3):
    '''
    preds: list of tensor (ch, h, w)

    return:
    processed_preds: list of tensor (ch, h, w)
    '''


    # first threshold the raw heatmaps
    if len(preds[0].shape) > 2:
        ix = np.argmin(preds[0].shape)
        if ix != 0:
            raise Exception("channels are not on the first dimension \
                            or are more than the spatial dimension")
        preds_t = [(np.squeeze(x[0:1, :, :]) > th) for x in preds]

    # then post-process by removing holes, small objects, watershed, and finally remove small objects introduced in last step
    processed_preds = []
    for p in tqdm(preds_t):
        labels_pred, nlabels_pred = ndimage.label(p)
        processed = remove_small_holes(labels_pred, area_threshold=area_threshold, connectivity=1)
        processed = remove_small_objects(processed, min_size=min_obj_size, connectivity=1)
        labels_bool = processed.astype(bool)
        distance = ndimage.distance_transform_edt(processed)

        maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
        local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones((foot, foot)),
                                    exclude_border=False,
                                    labels=np.squeeze(labels_bool))
        local_maxi = remove_small_objects(local_maxi, min_size=min_obj_size, connectivity=1)
        markers = ndimage.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=np.squeeze(labels_bool),
                           compactness=1, watershed_line=True)

        processed = remove_small_holes(labels, area_threshold=area_threshold, connectivity=1)
        processed = remove_small_objects(processed, min_size=50, connectivity=1)
        processed_preds.append(processed.astype("uint8")*255)
    return processed_preds


def get_prelabeled_masks(images_path:Path, th: float, batch_size: int, model_path: Path=MODEL_PATH):
    # load pretrained model
    model = torch.nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=1, c0=True,
                                      pretrained = False, progress= True)).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        model.load_state_dict(torch.load(model_path)['model_state_dict'])


    # dataloader
    cells_images = CellsLoader(images_path)
    data_loader = DataLoader(cells_images, batch_size=batch_size, shuffle=False, num_workers=6)
    filenames = cells_images.imgs_list

    # inference
    model.eval()
    preds = []
    for x in tqdm(data_loader, leave=True):
        with torch.no_grad():
            results = model(x.to(device)).cpu().detach()
            preds.extend(results)
            torch.cuda.empty_cache()

    preds_processed = post_processing(preds, th)

    masks_path: Path = images_path.parent / f'masks/{th}'
    masks_path.mkdir(exist_ok=True, parents=True)
    for outmask, fn in tqdm(zip(preds_processed, filenames)):
        io.imsave(masks_path / fn.name, outmask, check_contrast=False)  # , as_gray=True)


if __name__ == '__main__':
    images_path: Path = DATA_PATH / 'images'
    th: float = 0.55
    batch_size: int = 4
    get_prelabeled_masks(images_path=images_path, th=th, batch_size=batch_size)