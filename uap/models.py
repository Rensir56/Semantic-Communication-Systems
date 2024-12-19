import torch,torchvision
from torchvision import transforms
from datasets import get_transforms
transform_ImageNet=transforms.Compose([
    transforms.Normalize((123.675/255, 116.28/255, 103.53/255), (58.395/255, 57.12/255, 57.375/255)),
    
])
def get_model(model_name):
    if model_name=="ResNet_18":
        model=torchvision.models.resnet18(pretrained=True)
    elif model_name=="VGG_11":
        model=torchvision.models.vgg11_bn(pretrained=True)
    elif model_name=="DenseNet_121":
        model=torchvision.models.densenet121(pretrained=True)
    elif model_name=="Inception_V3":
        model=torchvision.models.inception_v3(pretrained=True)
    elif model_name=="EfficientNet_B2":
        model=torchvision.models.efficientnet_b2(pretrained=True)
    elif model_name=="GoogLeNet":
        model=torchvision.models.googlenet(pretrained=True)
    elif model_name=="ResNet_50":
        model=torchvision.models.resnet50(pretrained=True)
    elif model_name=="VGG_19":
        model=torchvision.models.vgg19(pretrained=True)
    elif model_name=="MobileNet_V2":
        model=torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name=="Wide_ResNet50_2":
        model=torchvision.models.wide_resnet50_2(pretrained=True)
    elif model_name=="Squeezenet1_0":
        model=torchvision.models.squeezenet1_0(pretrained=True)
    elif model_name=="Shufflenet_v2_x2_0":
        model=torchvision.models.shufflenet_v2_x2_0(pretrained=True)
    elif model_name=="MNASNet1_0":
        model=torchvision.models.mnasnet1_0(pretrained=True)
    elif model_name=="VIT_b_16":
        model=torchvision.models.vit_b_16(pretrained=True)
    elif model_name=="Convnext_tiny":
        model=torchvision.models.convnext_tiny(pretrained=True)
    else:
        print(model_name)
        raise NotImplementedError
    return model

class Mymodel(torch.nn.Module):
    def __init__(self,model_name,dataset_name="ImageNet",rand_init=False):
        super().__init__()
        self.model=get_model(model_name) # weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        self.transforms=get_transforms(dataset_name)
        self.model_name=model_name

        if rand_init:
            self.model.apply(self.kaiming_init)
        
        import json
        self.class_idx = json.load(open("/storage/Jie_Wan/uap/imagenet_class_index.json"))


    def forward(self, x):
        x=self.transforms(x)
        x=self.model(x)
        return x
    
    def predict_label(self,x):
        x=self.forward(x)
        x=torch.argmax(x,dim=1)
        return x
    
    def predict_label_from_confidence_vector(self,confidence_scores,require_label=False):
        top5_scores, top5_indices = torch.topk(confidence_scores, 5, dim=1)
        if require_label:
            total_top5_classes=[]
            for i in range(len(confidence_scores)):
                top5_classes=[]
                for index in top5_indices[i]:
                    top5_classes.append(self.class_idx[str(index.item())][1])
                # print(f"Top 5 classes for sample {i}: {top5_classes}")
                total_top5_classes.append(top5_classes)
            return top5_scores, total_top5_classes,top5_indices
        else:
            return top5_scores
    
    def kaiming_init(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

