from torch import  nn
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Resize
import os
class VGG_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = [2,2,3,3,3]
        self.conv_1_1 = nn.Conv2d(1,64,3,stride=1,padding=1)
        self.conv_1_2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.conv_2_1 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.conv_2_2 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.conv_3_1 = nn.Conv2d(128,256,3,stride=1,padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.adda = nn.AdaptiveMaxPool2d((7,7))
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2,2))
        self.drop = nn.Dropout(0.5)

    def forward(self,x):
        x = self.relu(self.conv_1_1(x))
        x = self.relu(self.conv_1_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_2_1(x))
        x = self.relu(self.conv_2_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_3_1(x))
        x = self.relu(self.conv_3_2(x))
        x = self.relu(self.conv_3_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_4_1(x))
        x = self.relu(self.conv_4_2(x))
        x = self.relu(self.conv_4_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv_5_1(x))
        x = self.relu(self.conv_5_2(x))
        x = self.relu(self.conv_5_3(x))
        x = self.adda(x)

        x = x.view(x.size(0),-1)

        x = self.relu(self.fc6(x))
        x = self.drop(x)
        x = self.relu(self.fc7(x))
        x = self.drop(x)
        
        return self.relu(self.fc8(x))

class custom_dataset(Dataset):
    def __init__(self,folderpath):
        self.img_path = folderpath

        self.file_list = os.listdir(folderpath)
    #It should return one of all images togather with the individual ID and the imageID


    def __len__(self):
        return self.file_list.__len__()

    def __getitem__(self, idx):
        name = self.file_list[idx]
        data = Image.open(self.img_path + name)
        re = Resize((128, 256))
        trans = ToTensor()
        return trans(re(data))       

        #Here we should return (anchor, positive,negative as a tuple)



class network():
    def __init__(self):
        self.model = VGG_encoder()




    def train(self,folderpath,epochs=3,):
        self.model.train()

        train_set = custom_dataset(folderpath)
        train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        for epoch in range(epochs):
            cum_loss = 0

            for ii,im in enumerate(train_loader):
                anchor = self.model(im)
                positive = self.model(im_pos)
                negative = self.model(im_neg)

                loss = triplet_loss(anchor, positive, negative)


        return



    def test(self):
        self.model.eval()
        return

        
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("folderpath",help="Path to train/test folder")
    argp.add_argument("-m","--mode",default="train",help="train/test/single")
    

    sys.argv = ["auto_encoder.py",
        "./train2/",
        "-m",
        "train"]


    args = vars(argp.parse_args())

    net = network()

    if args["mode"] == "train":
        net.train(args["folderpath"])
    elif args["mode"] == "test":
        net.test(args["folderpath"])
    elif args["mode"] == "single":
        net.single(args["folderpath"])
    else:
        print("Incorrect Mode")