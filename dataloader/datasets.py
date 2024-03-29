import os

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if args.data_set == 'IMNET':
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                no_aug=args.no_aug,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

            if args.data_set == 'CIFAR' and not args.no_aug:
                del transform.transforms[3]
                #transform.transforms[4] = transforms.Normalize()

        else:
            transform = transforms.Compose([
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(args.input_size, padding=0),
                transforms.ToTensor(),
            ])

        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.data_set == 'IMNET':
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
