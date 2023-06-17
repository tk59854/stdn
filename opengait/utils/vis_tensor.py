import torch
import torchvision.utils 

def show_tensor(tensor, num_img=4, num_channel=16, save_name='img', save=True, random=False):
    """
    tensor: [n, s, c, h, w]
    img: num_img raws and num_channel columns, from num_img imgs
    """
    if tensor.dim() == 5:
        _, s, c, h, w = tensor.shape
        img = tensor.view(-1, c, h, w).clone().detach().cpu().type(torch.float32)
    else:
        n, c, h, w = tensor.shape
        img = tensor.clone().detach().cpu().type(torch.float32)
    if random:
        img_idx = torch.randperm(img.shape[0])[:num_img]
        channel_idx = torch.randperm(img.shape[1])[:num_channel]
    else:
        img_idx = torch.arange(0, num_img)
        channel_idx = torch.arange(0, num_channel)
    img = img[img_idx, :, :, :]
    img = img[:, channel_idx, :, :].view(num_img*num_channel  , 1, h, w)
    img = torchvision.utils.make_grid(img, nrow=num_channel, padding=0, normalize=True, scale_each=True)
    
    if save:
        torchvision.utils.save_image(img, '{}.png'.format(save_name))
    else:
        return img


def show_tensor_3d(tensor, num_seqs=10, num_channel=16, channel_avg=False, save_name='img'):
    """
    tensor: [n, c, s, h, w]
    img: num_channel raws and num_seqs columns, all from one seq
    """

    _, c, s, h, w = tensor.shape
    img_idx = torch.randint(_, size=(1,))
    img = tensor[img_idx].clone().detach().cpu().squeeze(0).type(torch.float32)

    channel_idx = torch.randperm(img.shape[0])[:num_channel]

    if not channel_avg:
        img = img[channel_idx, :num_seqs, :, :].view(-1, 1, h, w)
    else:
        img = (torch.mean(img, 0, keepdim=True) + torch.max(img, 0, keepdim=True)[0]).view(-1, 1, h, w)
        # img = (torch.mean(img, 0, keepdim=True)).view(-1, 1, h, w)
    
    
    img = torchvision.utils.make_grid(img, nrow=num_seqs, normalize=True, scale_each=True)
    
    torchvision.utils.save_image(img, '{}.png'.format(save_name))

if __name__ == '__main__':
    x = torch.randn(4, 30, 32, 16, 16)
    y = show_tensor_3d(x)
