from torch.utils.data import DataLoader
from torchvision import transforms

def get_transform(args, split):
    scale_size = args.scale_size
    crop_size = args.crop_size
    
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((scale_size, scale_size)),
            transforms.RandomChoice([
                transforms.RandomCrop(640),
                transforms.RandomCrop(576),
                transforms.RandomCrop(512), 
                transforms.RandomCrop(384),
                transforms.RandomCrop(320)
            ]),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((scale_size, scale_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def get_loader(dataset, args, split):
    transform = get_transform(args, split)
    dataset.transform = transform
    
    shuffle = True if split == 'train' else False
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=shuffle
    )

    return data_loader