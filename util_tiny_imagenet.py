import os
import torch
from torchvision import datasets, transforms
from PIL import Image
import urllib.request
import zipfile
import tempfile

# --------------------------
# Tiny-ImageNet helper
# --------------------------
class TinyImageNetValDataset(torch.utils.data.Dataset):
    """
    Support the original Tiny-ImageNet-200 val layout:
      val/
        images/*.JPEG
        val_annotations.txt  (image_name \t wnid \t x y w h ...)
    If your val/ is already reorganized into class folders, just use ImageFolder.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        anno = os.path.join(root, 'val', 'val_annotations.txt')
        with open(anno, 'r') as f:
            lines = [x.strip().split('\t') for x in f.readlines()]
        # build wnid -> idx from train folder
        train_dir = os.path.join(root, 'train')
        wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        self.wnid_to_idx = {w:i for i, w in enumerate(wnids)}
        self.items = []
        for parts in lines:
            img, wnid = parts[0], parts[1]
            if wnid in self.wnid_to_idx:
                self.items.append((os.path.join(root, 'val', 'images', img), self.wnid_to_idx[wnid]))
        assert len(self.items) > 0, "No validation items found. Check Tiny-ImageNet path."

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def _download_and_prepare_tiny_imagenet(target_root: str):
    """
    Download Tiny-ImageNet-200 if missing and extract to target_root.
    Expected layout after extraction: target_root/{train, val, wnids.txt}
    """
    os.makedirs(target_root, exist_ok=True)
    # If already prepared, return
    expected_train = os.path.join(target_root, "train")
    expected_val = os.path.join(target_root, "val")
    if os.path.isdir(expected_train) and os.path.isdir(expected_val):
        return

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(target_root, "tiny-imagenet-200.zip")
    try:
        print(f"[Tiny-ImageNet] Downloading from {url} to {zip_path} ...")
        urllib.request.urlretrieve(url, zip_path)
        print("[Tiny-ImageNet] Download finished. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_root)
        # If extracted into a nested folder, move its content up to target_root
        nested = os.path.join(target_root, "tiny-imagenet-200")
        if os.path.isdir(nested):
            # Move contents up one level
            for name in os.listdir(nested):
                src_p = os.path.join(nested, name)
                dst_p = os.path.join(target_root, name)
                if not os.path.exists(dst_p):
                    import shutil as _shutil
                    _shutil.move(src_p, dst_p)
            # remove empty nested dir
            try:
                os.rmdir(nested)
            except OSError:
                pass
        print("[Tiny-ImageNet] Extraction done.")
    except Exception as e:
        print(f"[Tiny-ImageNet] Auto-download failed: {e}")
        print("Please manually download tiny-imagenet-200.zip from http://cs231n.stanford.edu/ and unzip to:", target_root)
        raise
    finally:
        # Keep the zip as cache; comment the following line out if you'd like to remove it automatically.
        pass

def build_tiny_imagenet_dataset(is_train, args):
    """
    Returns a dataset for Tiny-ImageNet-200.
    - Expects directory: args.tiny_imagenet_path (contains 'train' and 'val').
    - If val/ has 'val_annotations.txt' (original layout), use TinyImageNetValDataset.
      Otherwise assume val/ is reorganized as ImageFolder.
    - We upsample to 224 to fit ViT 224 models.
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
    if is_train:
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        ds = datasets.ImageFolder(os.path.join(args.tiny_imagenet_path, 'train'), transform=tfm)
    else:
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        # support both original and reorganized val structures
        anno_path = os.path.join(args.tiny_imagenet_path, 'val', 'val_annotations.txt')
        if os.path.exists(anno_path):
            ds = TinyImageNetValDataset(args.tiny_imagenet_path, transform=tfm)
        else:
            ds = datasets.ImageFolder(os.path.join(args.tiny_imagenet_path, 'val'), transform=tfm)
    return ds