from pathlib import Path

import skimage
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

class CellsLoader(Dataset):
    def __init__(self, images_path: Path):
        self.imgs_path = images_path
        self.imgs_list = [*self.imgs_path.iterdir()]
        self.transform = T.Compose([T.Lambda(lambda x: x * 1. / 255), T.ToTensor()])

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img = skimage.io.imread(self.imgs_path / self.imgs_list[idx])
        # img = cv2.imread(str(self.imgs_path / self.imgs_list[idx]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img.float()
        # return img.astype(np.float32)
