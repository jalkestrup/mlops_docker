import click
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import ConvNet


@click.group()
def cli():
    pass

# Datapath
main_path = 'data/processed'

@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    print("Visualize models mapping of data into feature space")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = ConvNet()
    state_dict = torch.load(f'{model_checkpoint}')
    model.load_state_dict(state_dict)
    images = torch.unsqueeze( torch.load(f'{main_path}/train_images.pt'), dim=1)
    labels = torch.load(f'{main_path}/train_labels.pt')
    
    with torch.no_grad():
        outputs = model(images)
        mapped = model.penult_layer.numpy()
        print(mapped.shape)

    tsne = TSNE()
    mapped_tsne = tsne.fit_transform(mapped)
    plt.figure()
    plt.scatter(mapped_tsne[:,0], mapped_tsne[:,1], c=labels.numpy())
    plt.savefig('reports/figures/penultimate_layer_tsne.png', dpi=300)

cli.add_command(visualize)

if __name__ == "__main__":
    cli()