
from pylab import *
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
# import random
# import time
# from scipy.misc import imread
# from scipy.misc import imresize
# import matplotlib.image as mpimg
import os
# from scipy.ndimage import filters
# import urllib
import threading
import skimage
# from urllib.request import Request, urlopen, urlretrieve
import urllib.request
from matplotlib import pyplot as plt


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                # print("Error in timeout...")
                # print(e)
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

         

def get_data(datafile, target_folder):
    act = list(set([a.split("\t")[0] for a in open(datafile).readlines()]))
    testfile = urllib.request.URLopener()   
    #Note: you need to create the target folder first in order 
    #for this to work
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(datafile):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], os.path.join(target_folder,filename)), {}, 30)
                if not os.path.isfile(os.path.join(target_folder,filename)):
                    continue

                print(filename)
                i += 1    



def decodeImage(data, shape):
    #Gives us 1d array
    decoded = np.frombuffer(data, dtype=np.uint8)
    decoded = decoded.reshape(shape)
    return decoded




url = 'http://www.independent.org/images/bios_hirez/garcia_andy_700.jpg'
img_shape = (875,700,3)
img_file_name = "./garcia_andy_700.jpg"


if __name__ == '__main__':
    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)
    
    #image = skimage.io.imread(url)
    
    # req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
    # webpage = urllib.request.urlopen(url).read()
    # urlretrieve(req, img_file_name)
    # print("\n")
    # print("Webpage: ")
    # print(webpage)
    # print("\n")
    # print(type(webpage))
    # print(dir(webpage))
    # image = decodeImage(webpage, img_shape)
    # print(image.shape)
    # print(image[:10])
    # skimage.io.imshow(image)
    
    
    # testfile = urllib.request.URLopener()
    # timeout(testfile.retrieve, (url, img_file_name), {}, 30)
    # arr = skimage.io.imread(img_file_name)
    # skimage.io.imshow(arr)
    
    
    
    urllib.request.urlretrieve(url, img_file_name)
    image = skimage.io.imread(img_file_name)
    skimage.io.imshow(image)
    plt.show()
    os.remove(img_file_name)