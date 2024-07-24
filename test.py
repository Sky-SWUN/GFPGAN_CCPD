import torch
import torchvision

# data = torch.load('./rgb0.pt')
# data = data*0.5 + 0.5
# torchvision.utils.save_image(data, './rgb0.png', nrow=1)

for i in range(7):
    data = torch.load('./rgb{}.pt'.format(i))*0.5+0.5
    torchvision.utils.save_image(data, './rgb{}.png'.format(i), nrow=1)
    # print(data.shape)