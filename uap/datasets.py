import torch
import torchvision

from torchvision import transforms,datasets

from torch.utils.data import DataLoader

def get_cifar10():
    transform1=transforms.Compose([
        transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    transform2=transforms.Compose([
        transforms.ToTensor(),
    ])
    train_datasets=datasets.CIFAR10(root="/data/home/public",train=True,download=False ,transform=transform1)
    val_datasets=datasets.CIFAR10(root="/data/home/public",train=False,download=False ,transform=transform2)
    
    return train_datasets,val_datasets
    
def get_cifar100():
    transform1=transforms.Compose([
        transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    transform2=transforms.Compose([
        transforms.ToTensor(),
    ])
    train_datasets=datasets.CIFAR100(root="/data/home/Jie_Wan/datasets",train=True,download=False ,transform=transform1)
    val_datasets=datasets.CIFAR100(root="/data/home/Jie_Wan/datasets",train=False,download=False ,transform=transform2)
    
    return train_datasets,val_datasets


def get_imagenet():
    transform1=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        
    ])
    transform2=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
       
        
    ])
    train_datasets=datasets.ImageNet(root="/data/home/public/source_datasets/ImageNet",split="train",transform=transform1)
    val_datasets=datasets.ImageNet(root="/data/home/public/source_datasets/ImageNet",split="val",transform=transform2)
    
    return train_datasets,val_datasets

def get_transforms(name):
    if name=="CIFAR10":
        transform=transforms.Compose([
        transforms.Normalize((125.307/255, 122.961/255, 113.8575/255), (51.5865/255, 50.847/255, 51.255/255)),
        ])
        
    elif name=="CIFAR100":
        transform=transforms.Compose([
        transforms.Normalize((129.304/255, 124.070/255, 112.434/255), (68.170/255, 65.392/255, 70.418/255)),
        ])
    elif name=="ImageNet":
        transform=transforms.Compose([
        transforms.Normalize((123.675/255, 116.28/255, 103.53/255), (58.395/255, 57.12/255, 57.375/255)),
        
    ])
    return transform

def get_ori_datasets(name):
    if name=="CIFAR10":
        _,val_datasets=get_cifar10()
        
    elif name=="CIFAR100":
        _,val_datasets=get_cifar100()
        
    elif name=="ImageNet":
        _,val_datasets=get_imagenet()

    return val_datasets


def get_attack_datasets(model,dataset_name,max_num=100,shuffle=False):
    dataset=get_ori_datasets(dataset_name)
    dataset_loader=DataLoader(dataset,50,shuffle=shuffle,num_workers=4)
    pre_label=[]
    ori_label=[]
    x_data=[]

    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(dataset_loader):
            
            x_data+=[i.unsqueeze(0).clone().detach().cpu() for i in x]

            x=x.cuda()
            logits=model(x)
            pre_y=torch.argmax(logits,dim=1).cpu()

            pre_label+=[i.item() for i in pre_y]
            ori_label+=[i for i in y]
    
    
    x_data=torch.cat(x_data,dim=0)
    pre_label=torch.tensor(pre_label).detach().cpu()
    ori_label=torch.tensor(ori_label).detach().cpu()

    print("acc is ",torch.mean((pre_label==ori_label).float()).item()*100)

    x_data=x_data[pre_label==ori_label]
    label=pre_label[pre_label==ori_label]

    # length=x_data.shape[0]

    x_pre,x_lateer=x_data[0:2000],x_data[2000:4000]
    label_pre,label_lateer=label[0:2000],label[2000:4000]

    x_pre,x_lateer=x_pre[label_pre!=label_lateer],x_lateer[label_pre!=label_lateer]
    label_pre,label_lateer=label_pre[label_pre!=label_lateer],label_lateer[label_pre!=label_lateer]

    assert x_pre.shape[0]>=max_num and x_lateer.shape[0]>=max_num and label_pre.shape[0]>=max_num and label_lateer.shape[0]>=max_num
    
    
    return CustomDataset(x_pre[:max_num],x_lateer[:max_num],label_pre[:max_num],label_lateer[:max_num])



def get_attack_datasets_tensor(model,dataset_name,max_num=100,shuffle=False):
    dataset=get_ori_datasets(dataset_name)
    dataset_loader=DataLoader(dataset,50,shuffle=shuffle,num_workers=4)
    pre_label=[]
    ori_label=[]
    x_data=[]

    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(dataset_loader):
            
            x_data+=[i.unsqueeze(0).clone().detach().cpu() for i in x]

            x=x.cuda()
            logits=model(x)
            pre_y=torch.argmax(logits,dim=1).cpu()

            pre_label+=[i.item() for i in pre_y]
            ori_label+=[i for i in y]
    
    
    x_data=torch.cat(x_data,dim=0)
    pre_label=torch.tensor(pre_label).detach().cpu()
    ori_label=torch.tensor(ori_label).detach().cpu()

    print("acc is ",torch.mean((pre_label==ori_label).float()).item()*100)

    x_data=x_data[pre_label==ori_label]
    label=pre_label[pre_label==ori_label]

    len_x= x_data.shape[0]
    
    assert len_x>=6000
    
    return Tensor_dataset(x_data[:5000],label[:5000]),Tensor_dataset(x_data[-1000:],label[-1000:]),torch.mean((pre_label==ori_label).float()).item()*100


def get_random_datasets_tensor(dataset_name,shuffle=False,num=5000):
    dataset=get_ori_datasets(dataset_name)
    dataset_loader=DataLoader(dataset,50,shuffle=shuffle,num_workers=4)
    pre_label=[]
    ori_label=[]
    x_data=[]

    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(dataset_loader):
            
            x_data+=[i.unsqueeze(0).clone().detach().cpu() for i in x]


            ori_label+=[i for i in y]

            if len(x_data)>=num*1.2:
                break
    
    
    x_data=torch.cat(x_data,dim=0)
    
    ori_label=torch.tensor(ori_label).detach().cpu()



    len_x= x_data.shape[0]
    
    assert len_x>=num*1.2
    
    return Tensor_dataset(x_data[:num],ori_label[:num]),Tensor_dataset(x_data[-num//5:],ori_label[-num//5:])



class CustomDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,x1,x2,y1,y2):
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
        self.len=self.x1.shape[0]

    def __getitem__(self, index):
        return  self.x1[index],self.x2[index],self.y1[index],self.y2[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.len
    


class Tensor_dataset(torch.utils.data.Dataset):
    def __init__(self, data,label, transform=None):
        super().__init__()
        self.data = data
        self.label= label

    def __len__(self):
        return self.data.shape[0]  # 数据集长度
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]







