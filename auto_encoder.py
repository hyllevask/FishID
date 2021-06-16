from torch import  nn
import torch
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Resize
import os
from numpy.random import randint


class VGG_encoder(nn.Module):
    def __init__(self):
        #VGG-Facenet

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

        self.adda = nn.AdaptiveMaxPool2d((7,7))     #Added adaptive pooling
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

        subfolderlist = os.listdir(self.img_path)
        self.file_list = []
        self.folder_list = []
        for subfolder in subfolderlist:

            im_path = os.listdir(self.img_path + "/" + subfolder)
            self.file_list.extend(im_path)
            self.folder_list.extend([self.img_path + subfolder + "/" for s in im_path] )    
            #print(self.file_list)
            #print(self.ID_list)

    #It should return one of all images togather with the individual ID and the imageID


    def __len__(self):
        return self.file_list.__len__()

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        foldername = self.folder_list[idx]

        full_path = foldername + "/" + filename
        data_a = Image.open(full_path)
        
        pos_list = os.listdir(foldername)
        
        while True:
            pos_int = randint(0,pos_list.__len__())
            pos_filename = pos_list[pos_int]
            if pos_filename != filename:
                break
        pos_fullpath = foldername + "/" + pos_filename
        data_p = Image.open(pos_fullpath)


        while True:
            neg_int = randint(0,self.__len__())
            neg_folder = self.folder_list[neg_int]
            neg_file = self.file_list[neg_int]

            if neg_folder !=foldername:
                break
        neg_fullpath = neg_folder + "/" + neg_file
        data_n = Image.open(neg_fullpath)


        re = Resize((128, 256))
        trans = ToTensor()
        return (trans(re(data_a)),trans(re(data_p)), trans(re(data_n)))       

        #Here we should return (anchor, positive,negative as a tuple)



class network():
    def __init__(self):
        self.model = VGG_encoder()
        if torch.cuda.is_available():  
            self.device = "cuda:0" 
        else:  
            self.device = "cpu" 
        self.model.to(torch.device(self.device))



    def train(self,folderpath,epochs=3,):
        self.model.train()

        train_set = custom_dataset(folderpath)
        train_loader = DataLoader(train_set,batch_size=4,shuffle=True)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        optimizer = optim.Adam(self.model.parameters(),lr = 0.01)

        for epoch in range(epochs):
            cum_loss = 0
            for ii,(im,im_pos,im_neg) in enumerate(train_loader):
                #print(ii)
                im = im.to(torch.device(self.device))
                im_pos = im_pos.to(torch.device(self.device))
                im_neg = im_neg.to(torch.device(self.device))
                optimizer.zero_grad()
                anchor = self.model(im)
                positive = self.model(im_pos)
                negative = self.model(im_neg)

                loss = triplet_loss(anchor, positive, negative)
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
                if ii % 20 == 19:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, ii + 1, cum_loss / 2000))
                    cum_loss = 0



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