import glob
import torch.utils
from torch.utils.data import Dataset,DataLoader
from torchvision  import transforms
import cv2
from PIL import Image
import torch
import random
class celecba256Dataset(Dataset):
    def __init__(self, max_length=5000 ):
        self.x_path=glob.glob("celeba-256/*/*")[:max_length]
        print(f"len of dataset is {len(self.x_path)}")
        self.transform = transforms.Compose([
            transforms.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224), # 从图片中间切出224*224的图片
            transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
        ])
        
        
        

    def __getitem__(self, item):
        image_path = self.x_path[item]

        image = Image.open(image_path)  # 读取到的是RGB， W, H, C
        image = self.transform(image)   # transform转化image为：C, H, W

        
        return image,0

    def __len__(self):
        return len(self.x_path)


class VOCDataset(Dataset):
    def __init__(self, max_length=5000 ):
        self.x_path=glob.glob("JPEGImages/*")[:max_length]
        print(f"len of dataset is {len(self.x_path)}")
        self.transform = transforms.Compose([
            transforms.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224), # 从图片中间切出224*224的图片
            transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
        ])
        
        
        

    def __getitem__(self, item):
        image_path = self.x_path[item]

        image = Image.open(image_path)  # 读取到的是RGB， W, H, C
        

        if len(image.size)==2:
            image=image.convert('RGB')

        image = self.transform(image)   # transform转化image为：C, H, W
        
        return image,0

    def __len__(self):
        return len(self.x_path)

class CocoDataset(Dataset):
    def __init__(self, max_length=5000):
        self.x_path=glob.glob("Coco/train2014/*")[:max_length]
        print(f"len of dataset is {len(self.x_path)}")
        self.transform = transforms.Compose([
            transforms.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224), # 从图片中间切出224*224的图片
            transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
        ])
        
        
        

    def __getitem__(self, item):
        image_path = self.x_path[item]

        image = Image.open(image_path)  # 读取到的是RGB， W, H, C
        

        if len(image.size)==2:
            image=image.convert('RGB')

        image = self.transform(image)   # transform转化image为：C, H, W
        
        return image,0

    def __len__(self):
        return len(self.x_path)


class JigsawDataset(Dataset):
    def __init__(self, max_length=5000):
        self.length=max_length
        print(f"len of dataset is {max_length}")
        self.transform = transforms.Compose([
            transforms.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224), # 从图片中间切出224*224的图片
            # transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 常用标准化
        ])
        
        
        

    def __getitem__(self, index):
        image=get_jigsaw([1,3,224,224])

        image = self.transform(image)   # transform转化image为：C, H, W
        
        return image,0

    def __len__(self):
        return self.length




def get_ranlist(num,min=0,max=225):
    list = [min]
    for i in range(num):
        x = random.randrange(min,max,step=1)
        list.append(x)
    return sorted(list)

def shuffle(img,wide=5,high=7,min=0,max=256,bound=224):
    #assert mode in [0, 1], 'check shuffle mode'

    wide_list = get_ranlist(wide,max=bound+1)
    high_list = get_ranlist(high,max=bound+1)
    for i in range(len(wide_list)):
        w_start = wide_list[i]
        if i < len(wide_list)-1:
            w_end = wide_list[i + 1]
        else:
            w_end = bound
        for j in range(len(high_list)):
            h_start = high_list[j]
            if j <len(high_list)-1:
                h_end = high_list[j+1]
            else:
                h_end = bound
            img[0, w_start:w_end, h_start:h_end] = random.randrange(min,max,1)
            img[1, w_start:w_end, h_start:h_end] = random.randrange(min,max,1)
            img[2, w_start:w_end, h_start:h_end] = random.randrange(min,max,1)
    return img


from skimage import filters
from skimage.morphology import disk
def get_jigsaw(shape,min=0,max=256,filter=False):
    # img_shape = torch.zeros_like(img.cpu().detach()).squeeze(0)
    # img_batch = torch.zeros_like(img.cpu().detach()).squeeze(0)
    img_shape = torch.zeros(shape).squeeze(0)
    img_batch = torch.zeros(shape).squeeze(0)

    
    #googlenet used the set of args.fre+2 nad args.fre for the jigsaw image
    ximg = shuffle(img_shape,)

    if filter == True:
        ximg = ximg.numpy()
        for i in range(len(ximg)):
            ximg[i] = filters.median(ximg[i], disk(5))
        ximg = torch.Tensor(ximg)
    # ximg = ximg.unsqueeze(0)  # .to(device)
    ximg = ximg / 255
        
    return ximg



if __name__=="__main__":
    from torchvision import utils as vutils
    def save_image_tensor(input_tensor: torch.Tensor, filename):
        """
        将tensor保存为图片
        :param input_tensor: 要保存的tensor
        :param filename: 保存的文件名
        """
        # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
        # 复制一份
        input_tensor = input_tensor.clone().detach()
        # 到cpu
        input_tensor = input_tensor.to(torch.device('cpu'))
        # 反归一化
        # input_tensor = unnormalize(input_tensor)
        vutils.save_image(input_tensor, filename)

    dataset=JigsawDataset(100)
    dataset=CocoDataset()
    dataset=VOCDataset()
    dataset=celecba256Dataset()
    # loader=DataLoader(dataset,10,False)

    # for i,(x,y) in enumerate(loader):
    #     save_image_tensor(x,"data_free_attack/jigsaw.png")