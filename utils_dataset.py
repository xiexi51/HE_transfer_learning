import os
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from locals import imagenet_root, data_root

def build_dataset(dataset_name, is_train, if_download, args):
    mean =  IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:
        if args.build_dataset_old and dataset_name != "imagenet":
            transform = transforms.Compose([
            transforms.RandomResizedCrop(224, antialias=True),
            # transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        else:
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
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    if dataset_name == "imagenet":
        dataset = datasets.ImageFolder(os.path.join(imagenet_root, 'train' if is_train else 'val'), transform=transform)
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=data_root, train=is_train, download=if_download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset string: {dataset_name}.")

    return dataset