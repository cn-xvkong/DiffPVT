import cv2
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):

        image_path = data_path[0]
        label_path = data_path[1]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        gt_np = np.array(label)
        edge = cv2.Canny(gt_np, 0, 255)
        edge = Image.fromarray(edge)
        # label_data = np.expand_dims(label, 0)

        return {
            "image": image,
            "label": label,
            "path": image_path,
            "edge": edge
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"数据读取出现问题，{self.datalist[i]}\n")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)

        if self.transform is not None:
            path = image["path"]
            image['image'] = self.transform(image['image'])
            image['mask'] = self.transform(image['label'])
            image['edge'] = self.transform(image['edge'])

        return {
            "image": image["image"],
            "mask": image["mask"],
            "path": path,
            "edge": image["edge"]
        }

    def __len__(self):
        return len(self.datalist)


def get_loader(data_dir, cache=False):
    all_train_images = sorted(glob.glob(f"{data_dir}/train/images/*.png"))
    all_train_labels = sorted(glob.glob(f"{data_dir}/train/masks/*.png"))

    all_test_images = sorted(glob.glob(f"{data_dir}/test/images/*.png"))
    all_test_labels = sorted(glob.glob(f"{data_dir}/test/masks/*.png"))

    all_train_paths = [[all_train_images[i], all_train_labels[i]] for i in range(len(all_train_images))]

    all_test_paths = [[all_test_images[i], all_test_labels[i]] for i in range(len(all_test_images))]

    train_files = []
    for p in all_train_paths:
        train_files.append(p)

    test_files = []
    for p in all_test_paths:
        test_files.append(p)

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    train_ds = PretrainDataset(train_files, transform=transform, cache=cache)

    val_ds = PretrainDataset(test_files, transform=transform, cache=cache)

    test_ds = PretrainDataset(test_files, transform=transform)

    loader = [train_ds, val_ds, test_ds]

    return loader
