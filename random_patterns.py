import numpy as np
from numpy.core.fromnumeric import shape
import png
from numpy.random import rand, randn, randint
import os
import argparse
import sys

class random_pattern():
    def __init__(self,num_spots,size,mean_diameter,std):
        self.size = size
        self.mean_diameter = mean_diameter
        self.std = std
        self.num_spots = num_spots
    def generate_images(self,N,folder_path):
        #Generates N number of images with the current properties

        #Check if folder exsists, otherwise create it 
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        for ii in range(N):
            im = np.zeros(self.size)
            im = im == 1
            #Sample positions
            x = randint(0,self.size[1],size = self.num_spots)
            y = randint(0,self.size[0],size = self.num_spots)

            #Sample size
            r = randn(self.num_spots)*self.std + self.mean_diameter/2
            YY,XX = np.mgrid[0:self.size[0],0:self.size[1]]

            for jj in range(self.num_spots):
                im_temp = (YY-y[jj])**2 + (XX-x[jj])**2 < r[jj]**2
                im = im | im_temp
            im = im*127
            
            r,c = shape(im)
            im2 = np.floor(im + randn(r,c)*10)+100
            im3 = np.floor(im + randn(r,c)*20)+100

            im = im.astype(np.int8)
            im2 = im2.astype(np.int8)
            im3 = im3.astype(np.int8)

            if not os.path.isdir(folder_path + "/ID" + str(ii)):
                os.mkdir(folder_path + "/ID" + str(ii))
            png.from_array(im,mode='L').save(folder_path + "/ID" + str(ii) + "/img_0.png")
            png.from_array(im2,mode='L').save(folder_path + "/ID" + str(ii) + "/img_1.png")
            png.from_array(im3,mode='L').save(folder_path + "/ID" + str(ii) + "/img_2.png")



argp = argparse.ArgumentParser()
argp.add_argument("folder_path")
argp.add_argument("--N_ind",type=int,default=10)
argp.add_argument("--size", type=int,default=(256,512))
argp.add_argument("--mean", type=int,default=20)
argp.add_argument("--std", type=int,default=5)
argp.add_argument("--num_spots",type=int,default=50)
#if __name__=='__main__':
#sys.argv = ['random_patterns.py', './test', '--N_ind', '10']
args = vars(argp.parse_args())


test = random_pattern(args['num_spots'],args['size'],args['mean'],args['std'])
test.generate_images(args['N_ind'],args['folder_path'])

print("end")