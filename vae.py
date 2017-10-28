
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.nn import functional as F
#internals
import gen_plot as gp

latent_dim = 2 # z size
mb_size = 32   # minibatch size

train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.ToTensor()),
                batch_size = mb_size, shuffle=True)

class vae(nn.Module):
    def __init__(self, latent_dim):
        super(vae, self).__init__()

        self.latent_dim = latent_dim
        # inference network
        self.input_hidden1 = nn.Linear(784, 200)
        self.hidden1_hidden2 = nn.Linear(200, 200)
        self.hidden2_mu = nn.Linear(200, latent_dim)
        self.hidden2_sig = nn.Linear(200, latent_dim)
        # generation network
        self.z_hidden3 = nn.Linear(latent_dim, 200)
        self.hidden3_hidden4 = nn.Linear(200, 200)
        self.hidden4_output = nn.Linear(200, 784)

    def Q(self, x):
        # inference network
        hidden1 = F.relu(self.input_hidden1(x.view(-1, 784)))
        hidden2 = F.relu(self.hidden1_hidden2(hidden1))
        z_mu = self.hidden2_mu(hidden2)
        z_logsig = self.hidden2_sig(hidden2)
        return z_mu, z_logsig

    def P(self, z):
        # generation network
        hidden3 = F.relu(self.z_hidden3(z))
        hidden4 = F.relu(self.hidden3_hidden4(hidden3))
        y  = F.sigmoid(self.hidden4_output(hidden4))
        return y

    def forward(self, x):
        # forward pass
        # inference network generates latent parameters (mean and log variance)
        z_mu, z_logsig = self.Q(x)
        # reparametrization trick to differentiate z_sample
        z_prior = Variable(torch.randn(1, latent_dim)).cuda()
        std = z_logsig.mul(0.5).exp_()
        z_sample = z_prior.mul(std).add_(z_mu)
        y = self.P(z_sample)
        return y, z_mu, z_logsig

def criterion(y, x, z_mu, z_logsig):
    # recontruction loss
    recon = F.binary_cross_entropy(y, x.view(-1, 784))
    # kl divergence between Q(z|x) ~ N(mu, sigma) and P(z) ~ N(0, I)
    kl = -0.5 * torch.sum(1 + z_logsig - z_mu.pow(2) - z_logsig.exp()) / (mb_size * 784)
    return recon + kl

model = vae(latent_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
for e in range(1, epochs+1):
    print('Start training.')
    print('Epoch = ' + str(e))
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).cuda()
        optimizer.zero_grad()
        y, mu, sig = model(data) # y = reconstructed image
        loss = criterion(y, data, mu, sig)
        loss.backward()
        optimizer.step()
    print(loss.data[0])

gp.latent_space_representation(20, (-10,10), model) # explore latent space representation
