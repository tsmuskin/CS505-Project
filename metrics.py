import torch
import torchvision.transforms as transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import glob
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])

generated_images = []
for f in glob.iglob("C:\\Users\\User\\Documents\\GitHub\\Metrics\\data\\generated\\*"):
    temp = Image.open(f)
    temp = temp.convert('RGB')
    temp = np.array(temp).astype(np.uint8)
    temp = temp.transpose(2,1,0)
    temp = torch.from_numpy(temp)
    generated_images.append(temp)
    
generated_images = torch.stack(generated_images, dim=0)
generated_images = generated_images.type(torch.uint8)

inception = InceptionScore()
inception.update(generated_images)
print(inception.compute())

real_images = []
for f in glob.iglob("C:\\Users\\User\\Documents\\GitHub\\Metrics\\data\\real\\*"):
    temp = Image.open(f)
    temp = temp.convert('RGB')
    temp = np.array(temp).astype(np.uint8)
    temp = temp.transpose(2,1,0)
    temp = torch.from_numpy(temp)
    real_images.append(temp)
    
real_images = torch.stack(real_images, dim=0)
real_images = real_images.type(torch.uint8)

fid = FrechetInceptionDistance(feature=64)
fid.update(generated_images, real=True)
fid.update(real_images, real=False)
print(fid.compute())