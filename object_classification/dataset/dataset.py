from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.svhn import SVHN


def load_imagenet(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    HR_size = 256
    tr_transform = transforms.Compose([
        transforms.RandomSizedCrop(HR_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Scale(int(256 * (HR_size / 224))),
        transforms.CenterCrop(HR_size),
        transforms.ToTensor(),
        normalize,
    ])
        
        
    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'imagenet', 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'imagenet', 'val'), transform=val_transform)
    return tr_dataset, val_dataset


def load_svhn(args):
    normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                     std=[0.19803012, 0.20101562, 0.19703614])

    tr_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
        
    tr_dataset = SVHN(os.path.join(args.data_dir, 'svhn'), split='train', transform=tr_transform, download=True)
    val_dataset = SVHN(os.path.join(args.data_dir, 'svhn'), split='test', transform=val_transform, download=True)
    return tr_dataset, val_dataset