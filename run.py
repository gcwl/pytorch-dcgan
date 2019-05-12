def main():
    import random
    from pathlib import Path
    import numpy as np

    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision.transforms import Compose, ToTensor, Normalize, Resize
    from torchvision import datasets

    from models import Generator, Discriminator, Gan
    from utils import get_yaml_config

    ROOT = Path(".")
    CONFIG = get_yaml_config(ROOT / "config.yml")
    DATA = ROOT / "data"
    DATA.mkdir(exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = datasets.CIFAR10(
        ROOT / "data",
        train=True,
        download=True,
        transform=Compose(
            [Resize(CONFIG.image_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    )

    dataloader = DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=True, drop_last=True)

    G = Generator(CONFIG.hidden_size, CONFIG.channel_size).to(DEVICE)
    D = Discriminator(CONFIG.image_size, CONFIG.channel_size).to(DEVICE)
    gan = Gan(CONFIG, dataloader, G, D, DEVICE)

    gan.train()


if __name__ == "__main__":
    main()
