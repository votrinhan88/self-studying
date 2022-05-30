import torch, torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
print(f'{__name__} - fix 13')

class Dataset:
    def __init__(self, batch_size = BATCH_SIZE):
        self.batch_size = batch_size

    def get_MNIST(self):
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,))
            ])

        # MNIST dataset
        train_set = torchvision.datasets.MNIST(
            root = './data/',
            train = True,
            # download = True,
            transform = transform
        )
        test_set = torchvision.datasets.MNIST(
            root = './data/',
            train = False,
            transform = transform
        )

        # Dataloaders
        train_loader = torch.utils.data.DataLoader(
            dataset = train_set,
            batch_size = self.batch_size,
            shuffle = True,
            worker_init_fn = self.worker_init_fn,
            generator = torch.Generator())
        test_loader = torch.utils.data.DataLoader(
            dataset = test_set,
            batch_size = self.batch_size*8,
            shuffle = True,
            worker_init_fn = self.worker_init_fn,
            generator = torch.Generator())
    
        return train_loader, test_loader

    def get_CIFAR10(self):
        train_set = torchvision.datasets.CIFAR10(
            root = './data/',
            train = True,
            # download = True,
            transform = T.Compose([
                T.RandomHorizontalFlip(p = 0.5),
                T.RandomRotation(degrees = 5),
                T.RandomCrop(size = (32, 32), padding = 3, padding_mode = 'symmetric'),
                T.ColorJitter(brightness    = 0.05,
                              contrast      = 0.05,
                              saturation    = 0.05,
                              hue           = 0.05),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],
                            std  = [0.229, 0.224, 0.225])]))
        test_set = torchvision.datasets.CIFAR10(
            root = './data/',
            train = False,
            download = True,
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406],
                            std  = [0.229, 0.224, 0.225])]))

        # Dataloaders
        train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                                   batch_size = self.batch_size,
                                                   shuffle = True,
                                                   worker_init_fn = self.worker_init_fn,
                                                   generator = torch.Generator())
        test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                                  batch_size = self.batch_size*8)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        return train_loader, test_loader, classes

    def worker_init_fn(self, worker_id):
        process_seed = torch.initial_seed()
        # Back out the base_seed so we can use all the bits.
        base_seed = process_seed - worker_id
        ss = np.random.SeedSequence([worker_id, base_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    @staticmethod
    def plot(raw_img, norm_img, trans_imgs, label, **imshow_kwargs):
        fig, axes = plt.subplots(nrows = 1, ncols = len(trans_imgs) + 2)

        axes[0].imshow(raw_img.permute(1, 2, 0))
        axes[0].set_title('Original image')

        axes[1].imshow(norm_img.permute(1, 2, 0))
        axes[1].set_title('Normalized image')
        # axes[([0, 1])].title.set_size(8)

        for col_idx, trans_img in enumerate(trans_imgs):
            col_idx = col_idx + 2
            axes[col_idx].imshow(trans_img.permute(1, 2, 0), **imshow_kwargs)


        fig.suptitle(label)
        plt.tight_layout()
        plt.show()

# Test train transformation
if __name__ == "__main__":
    import copy

    # Get datasets with & without transformation
    d = Dataset()
    train_loader, test_loader, classes = d.get_CIFAR10()
    dset_train = train_loader.dataset
    dset_train_raw = copy.copy(dset_train)
    dset_train_raw.transform = T.ToTensor()
    dset_train_norm = copy.copy(dset_train)
    dset_train_norm.transform = test_loader.dataset.transform

    # Get a random image from dataset
    idx = torch.randint(size = [1], high = dset_train.__len__())
    raw, label = dset_train_raw.__getitem__(idx)
    norm, _ = dset_train_norm.__getitem__(idx)
    transformed = [dset_train.__getitem__(idx)[0] for i in range(4)]
    d.plot(raw, norm, transformed, classes[label])