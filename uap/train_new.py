"""
Boundary Attack++
"""
import numpy as np
import torch
import os
class ExceedQueryLimitError(Exception):
    pass
query_time=0
from data_free_attack.data_freee_dataset import *
def UAP_ATTACK_WHITE(
    model_fns,
    train_data=None,
    uap=None,
    optimizer=None,
    alpha=1,
    args =None
):
    
    
    
    
    
    from tqdm import tqdm
    

    # beta=torch.ones(1).cuda()
    # gamma=torch.ones(1).cuda()
    # gamma=2-beta
    W = torch.ones(len(model_fns)).cuda() / len(model_fns)
    for epoch_i in range(1,args.epoch+1):
        print("\nepoch %d==>"%(epoch_i))
        print(W)

        

        

        with tqdm() as tbar:
            
            K = len(model_fns)
            
            for idx,(x,y) in enumerate(train_data):
                uap.requires_grad = True
                W.requires_grad = False

                x=x.cuda()
                y=y.cuda()

                
                loss_list=[]
                for j, model_fn in enumerate(model_fns):
                    logits_ori = model_fn(x)
                    logits_uap = model_fn(x+uap.repeat(1,1,alpha,alpha))
                    if args.loss == "CE":
                        loss = -W[j]*torch.nn.CrossEntropyLoss()(logits_uap,y).mean()
                    elif args.loss == "CE-tar":
                        loss = W[j]*torch.nn.CrossEntropyLoss()(logits_uap,torch.ones_like(y)*args.tar_label).mean()
                    elif args.loss == "Cosine-onehot-tar":
                        y_one_hot= torch.nn.functional.one_hot(torch.ones_like(y)*args.tar_label, num_classes= 1000) 
                        loss = -(torch.cosine_similarity(logits_uap,y_one_hot)).mean()
                    elif args.loss == "Cosine":
                        loss = W[j]*(torch.cosine_similarity(logits_ori,logits_uap)).mean()
                    elif args.loss == "Cosine-tar-with-unsupervised":
                        tar_zip=torch.load(f"exp/embedding_ori/{model_fn.model_name}.pth")
                        tar_embedding_index = tar_zip["train_label"]==0
                        # tar_embedding_index = tar_zip["train_label"]==109
                        tar_embedding = torch.mean(tar_zip["train_logit"][tar_embedding_index],dim=0,keepdim=True).cuda()
                        loss = W[j]*((torch.cosine_similarity(logits_ori,logits_uap)).mean()-(torch.cosine_similarity(tar_embedding,logits_uap)).mean())
                    elif args.loss == "Cosine-tar-without-unsupervised":
                        tar_zip=torch.load(f"exp/embedding_ori/{model_fn.model_name}.pth")
                        tar_embedding_index = tar_zip["train_label"]==0
                        # tar_embedding_index = tar_zip["train_label"]==109
                        tar_embedding = torch.mean(tar_zip["train_logit"][tar_embedding_index],dim=0,keepdim=True).cuda()
                        loss = W[j]*(-(torch.cosine_similarity(tar_embedding,logits_uap)).mean())
                    elif args.loss == "Cosine-onehot":
                        y_one_hot= torch.nn.functional.one_hot(y, num_classes= 1000) 
                        loss = W[j]*(torch.cosine_similarity(logits_uap,y_one_hot)).mean()
                    elif args.loss == "Cosine-partial":
                        values, indices = torch.topk(logits_ori, 10, dim=1)
                        partial_logits_ori = torch.gather(logits_ori, 1, indices)
                        partial_logits_uap = torch.gather(logits_uap, 1, indices)
                        loss = W[j]*(torch.cosine_similarity(partial_logits_ori,partial_logits_uap)).mean()
                    elif args.loss == "Cosine-onehot-new":
                        y_one_hot= torch.nn.functional.one_hot(y, num_classes= 1000) *2-1
                        loss = W[j]*(torch.cosine_similarity(logits_uap,y_one_hot)).mean()
                    elif args.loss == "Cosine-onehot-remove-groundtruth":
                        y_one_hot= torch.nn.functional.one_hot(y, num_classes= 1000) 
                        loss = (torch.cosine_similarity(logits_ori*(1-y_one_hot),logits_uap)).mean()
                    elif args.loss == "MSE":
                        loss = -W[j]*(torch.nn.MSELoss()(logits_ori,logits_uap)).mean()
                    elif args.loss == "MSE-onehot":
                        y_one_hot= torch.nn.functional.one_hot(y, num_classes= 1000) .float()
                        loss = -(torch.nn.MSELoss()(logits_uap,y_one_hot)).mean()
                    loss_list.append(loss.unsqueeze(0))
                final_loss=torch.cat(loss_list,dim=0).mean()#+(beta-1)**2
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
                
                

                uap.data = torch.clamp(uap.data, -(args.clip_value), args.clip_value)
                

                W.requires_grad = True
                uap.requires_grad = False

                loss_list=[]
                for j, model_fn in enumerate(model_fns):
                    if args.loss == "Cosine":
                        logits_uap = model_fn(x+uap.repeat(1,1,alpha,alpha).clone().detach())
                        loss = W[j]*(torch.cosine_similarity(logits_ori,logits_uap)).mean()
                    if args.loss == "Cosine-partial":
                        values, indices = torch.topk(logits_ori, 10, dim=1)
                        partial_logits_ori = torch.gather(logits_ori, 1, indices)
                        partial_logits_uap = torch.gather(logits_uap, 1, indices)
                        loss = W[j]*(torch.cosine_similarity(partial_logits_ori,partial_logits_uap)).mean()
                    # loss = W[j]*(-(torch.cosine_similarity(tar_embedding,logits_uap)).mean())
                    loss_list.append(loss.unsqueeze(0))
                loss_w = torch.cat(loss_list,dim=0).mean() - (K * (W - 1 / K).norm()).mean()

                grad_w = torch.autograd.grad(loss_w, W, retain_graph=False, create_graph=False)[0]
                W.data += args.lr_w * grad_w

                # Constrain W
                W = project_simplex(W)


                tbar.set_postfix()
                tbar.update(1)
            
            
        
        save_path = "exp/uap/{}/{}/{}/clip={},alpha={},lr={}".format(epoch_i,args.dataset,args.loss,args.clip_value,alpha,args.lr)
        os.makedirs(save_path,exist_ok=True)
        
        save_image_tensor(uap,os.path.join(save_path,"uap_single.png"))
        save_image_tensor(uap*10,os.path.join(save_path,"uap_single*10.png"))
        save_image_tensor(uap.repeat(1,1,alpha,alpha)+x,os.path.join(save_path,"adversarial.png"))
        save_image_tensor(x,os.path.join(save_path,"benign.png"))
        torch.save({"perturbation_hist":uap,"alpha":alpha,"w":W},os.path.join(save_path,"uap.pth"))
        
        
            
            
                
        

    
 





def project_simplex(v, z=1.0, axis=-1):
    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


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





def main():
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
        choices=["CIFAR10",  "CIFAR100", "ImageNet","celeba-256","Coco","VOC","Jigsaw"],
        default="ImageNet")
    parser.add_argument('--model_list', type=str,
        default='["GoogLeNet", "DenseNet_121","Inception_V3"]')
    parser.add_argument('--constraint', type=str,
        choices=['l2', 'linf'],
        default='l2')
    parser.add_argument('--attack_type', type=str,
        choices=['targeted', 'untargeted'],
        default='untargeted')
    parser.add_argument('--cuda', type=str,
        choices=["0","1","2","3"],
        default="0")
    parser.add_argument('--tar_label', type=str,
        default=0)
    parser.add_argument('--alpha', type=int,
        default=1)
    parser.add_argument('--epoch', type=int,
        default=5)
    parser.add_argument('--ramdom_num', type=int,
        default=10086)
    parser.add_argument('--clip_value', type=float,choices=[0.02,0.04,0.06,0.08],
        default=0.04)
    parser.add_argument('--lr', type=float,
        default=0.005)
    parser.add_argument('--lr_w', type=float,
        default=0.003)
    parser.add_argument('--batch_size', type=int,
        default=10)
    parser.add_argument('--loss', type=str,
        choices=['CE',"Cosine-onehot-new","Cosine-partial", 'CE-tar','Cosine',"Cosine-tar-with-unsupervised","Cosine-tar-without-unsupervised","Cosine-onehot","Cosine-onehot-tar","MSE","MSE-onehot","mix","mix-divide","mix-negative","Cosine-onehot-remove-groundtruth"],
        default='Cosine')
    args = parser.parse_args()
    args.model_list=eval(args.model_list)
    print(args)

    import os
    os.environ['CUDA_VISIBLE_DEVICES']=args.cuda
    # os.environ['CUDA_VISIBLE_DEVICES']="0,1"

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(args.ramdom_num)


    
    
    import sys
    sys.path.append(".") 
    from datasets import get_cifar10,get_cifar100,get_imagenet,get_attack_datasets_tensor,get_random_datasets_tensor
    from torch.utils.data import DataLoader
    from models import Mymodel
    model_list=[]
    for model_i in args.model_list:
        
        
        model=Mymodel(model_i)
        model.eval()
        model=model.cuda()

        model_list.append(model)
        
    if args.dataset in ["CIFAR10","CIFAR100"]:
        image_size = 32
    elif args.dataset in ["ImageNet","celeba-256","Coco","VOC","Jigsaw"]:
        image_size = 224
    else:
        raise NotImplementedError
    
    uap = (torch.rand(1,3,image_size//args.alpha,image_size//args.alpha).cuda()*2-1)*args.clip_value
    
    if args.dataset == "celeba-256":
        train_datasets = celecba256Dataset()
    elif args.dataset == "Coco":
        train_datasets = CocoDataset()
    elif args.dataset == "VOC":
        train_datasets = VOCDataset()
    elif args.dataset == "Jigsaw":
        train_datasets = JigsawDataset()
    elif args.dataset == "ImageNet":
        if not os.path.exists("processed_dataset/random.dataset"):
            os.makedirs("processed_dataset",exist_ok=True)
            print("start process dataset")
            train_datasets,val_datasets=get_random_datasets_tensor(args.dataset,shuffle=True)
            torch.save({"train_datasets":train_datasets,"val_datasets":val_datasets},"processed_dataset/random.dataset")
            print("end process dataset")
        else:
            print("start load dataset")
            dataset= torch.load("processed_dataset/random.dataset")
            train_datasets,val_datasets = dataset["train_datasets"],dataset["val_datasets"]
            print("end load random dataset")
    elif args.dataset == "CIFAR10":
        train_datasets,_ = get_cifar10()
    elif args.dataset == "CIFAR100":
        train_datasets,_ = get_cifar100()
    else:
        raise NotImplementedError

    print(len(train_datasets))

    train_data = DataLoader(train_datasets,args.batch_size,shuffle=True)
    # val_data = DataLoader(val_datasets,args.batch_size,shuffle=False)


    
    
    optimizer = torch.optim.Adam([uap], lr=args.lr,
                                          weight_decay=1e-5)
    

    UAP_ATTACK_WHITE(model_list,train_data,uap,optimizer,alpha=args.alpha,args=args)



if __name__=="__main__":
    main()


        
    
    