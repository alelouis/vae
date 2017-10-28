import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

def latent_space_representation(grid_w, extent, model):
  x = np.linspace(extent[0], extent[1], grid_w)
  y = x
  final_img_grid = np.zeros(shape=(28*grid_w, 28*grid_w))
  x_pixel, y_pixel = 0,0
  for i in x:
      for j in y:
          z = Variable(torch.FloatTensor([i, j])).cuda()
          sample = model.P(z).cpu()
          sample = sample.data.cpu().numpy().reshape(28,28)
          final_img_grid[x_pixel:x_pixel+28, y_pixel:y_pixel+28] = sample
          y_pixel += 28
      x_pixel += 28
      y_pixel = 0
  fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
  plt.imshow(final_img_grid, extent=[extent[0],extent[1],extent[0],extent[1]], cmap='gnuplot2') # get current figure
  plt.title('VAE latent space representation', fontsize = 14)
  plt.xlabel('z dimension 1', fontsize = 14)
  plt.ylabel('z dimension 2', fontsize = 14)
  plt.savefig("latent_space.png")
