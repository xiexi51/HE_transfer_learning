import os
from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from locals import imagenet_root


def build_imagenet_dataset(is_train, args):
    mean =  IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
    else:
        t = []
        t.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(224))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)

    imagenet_dataset = datasets.ImageFolder(os.path.join(imagenet_root, 'train' if is_train else 'val'), transform=transform)
    return imagenet_dataset
