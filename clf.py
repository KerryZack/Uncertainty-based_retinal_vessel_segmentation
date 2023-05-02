from torchvision import models, transforms
import torch
from PIL import Image
import time
from image import tiramisu
from image import stdunet
#import cv2
import numpy as np
import torch.nn as nn

def apply_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()

def get_uncertainty(data, model, T_samples):
    model.train()
    model.apply(apply_dropout)  # 在测试时同样启用dropout
    since = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = T_samples  # number of bayesian dropout samples
    outputs = torch.Tensor(np.array([model(data).cpu().detach().numpy() for _ in range(n_samples)])).to(device)
    # print(outputs.shape)

    # 得到方差
    std = torch.std(outputs, axis=0)
    # 得到均值
    mean = torch.mean(outputs, axis=0)

    time_elapsed = time.time() - since
    # print(time_elapsed)

    return mean, std


def predict(image_path):
    # if option =="resnet101":
    #     model = models.resnet101(pretrained=True)
    # elif option =="resnet50":
    #     model = models.resnet50(pretrained=True)
    # elif option == "densenet121":
    #     model = models.densenet121(pretrained=True)
    # elif option == "shufflenet_v2_x0_5":
    #     model = models.shufflenet_v2_x0_5(pretrained=True)
    # else:
    #     model = models.mobilenet_v2(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建Uncertainty network并加载权重
    pretrained_dict = torch.load('./checkpoint/checkpoint_chasefcn67.pth' ,map_location=device)
    fcn = tiramisu.FCDenseNet67(n_classes=1).to(device)
    model_dict = fcn.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    fcn.load_state_dict(model_dict)

    # 创建Segmentation network并加载权重
    sg_pretrained_dict = torch.load("./checkpoint/checkpoint_unetmean.pth",map_location=device )
    Sg_model = stdunet.build_unet(n_channels=3, n_classes=1).to(device)
    # sgmodel_dict = Sg_model.state_dict()
    # sg_pretrained_dict = {k: v for k, v in sg_pretrained_dict.items() if k in sgmodel_dict}
    # sgmodel_dict.update(sg_pretrained_dict)
    Sg_model.load_state_dict(sg_pretrained_dict['model_state_dict'])



    #https://pytorch.org/docs/stable/torchvision/models.html
    # transform = transforms.Compose([
    # transforms.Resize(512)
    # # # transforms.CenterCrop(224),
    # # transforms.ToTensor(),
    # # transforms.Normalize(
    # # mean=[0.485, 0.456, 0.406],
    # # std=[0.229, 0.224, 0.225]
    # )])

    # img = Image.open(image_path)
    image = Image.open("image/Image_01L.jpg")
    resize = transforms.Resize([512,512])
    image = resize(image)
    image = np.array(image)
    image = image / 255.0  ## (512, 512, 3)
    image = image[:, :, ::-1]
    image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)

    batch_t = image.unsqueeze(0).to(device)
    # batch_t = torch.unsqueeze(transform(image), 0)


    #get uncertainty




    Sg_model.eval()
    t1 = time.time()
    #Uncertainty
    mean, std = get_uncertainty(batch_t, fcn, 50)
    std = std.squeeze()
    print(std.shape) #512,512
    std = std.unsqueeze(0)
    std = std.unsqueeze(0)

    # print(std.shape)
    #prediction
    out = Sg_model(batch_t,std)
    t2 = time.time()

    fps = round(float(1 / (t2 - t1)), 3)

    #图像后处理
    mean = mean.squeeze()
    mean = torch.sigmoid(mean)*255
    mean = mean.cpu().detach().numpy()
    prior = np.where(mean>0.5,1,0)


    std = std.squeeze().cpu().numpy()


    # std = np.array(std*255,dtype=np.uint8)

    # std = Image.fromarray(std)
    # std = np.transpose(std, (2, 0, 1))
    # std = transforms.ToPILImage(std)

    predict = out.squeeze()
    predict = torch.sigmoid(predict)
    predict = predict.cpu().detach().numpy()
    mask = np.where(predict>0.5,0,1)

    return fps,prior,std,mask

    # with open('imagenet_classes.txt') as f:
    #     classes = [line.strip() for line in f.readlines()]
    # prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # _, indices = torch.sort(out, descending=True)

    # return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]],fps


