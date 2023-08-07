import os, sys
import torch
import skimage
import threading
import time
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from pylab import *
from matplotlib import pyplot as plt
from IPython.display import display
from pathlib import Path


opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)




def make_path(path:str):
    Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
    return path




def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None, verbose=False):
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
                if verbose:
                    print("\nError in Timeout:")
                    print(e)
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result




class FaceScrubDataset(Dataset):
    DTYPE = torch.float32
    def __init__(self, actors_file:str=None, actresses_file:str=None, target_image_size:int=28, 
                 target_folder:str="./uncropped", transform=None, train_size:int=70, get_data_at_first_call:bool=False,
                 delete_image_after_download:bool=True, verbose:bool=False):
        # super(Dataset, self).__init__()
        self.actors_file = actors_file
        self.actresses_file = actresses_file
        self.target_image_size = target_image_size
        self.target_folder = target_folder
        self.transform = transform
        self.train_size = train_size
        self.get_data_at_first_call = get_data_at_first_call
        self.testfile = urllib.request.URLopener()
        self.processed_files = []
        self.artists_counts = {}
        self.artists_list = []
        self._num_artists = 0
        self.train_indices = []
        self.test_indices = []
        self.delete_image_after_download = delete_image_after_download
        self.df = pd.DataFrame(
            columns=['name','name_idx','sex','var1','var2','url','x1','x2','y1','y2','hash','subset','is_available'])
        for int_col in ['name_idx','var1','var2','x1','x2','y1','y2','is_available']:
            self.df[int_col] = self.df[int_col].astype(np.int32)
        self.images = []
        self.labels = []
        self.data_is_processed = False
        self.files_are_processed = False
        self.image_cols = []
        self.data_cols = list(self.df.columns)
        if self.get_data_at_first_call:
            self.process_all_data(verbose=verbose)
        if verbose: print("Dataset initialized successfully.")
            
        
                
            
    def __len__(self):
        if not self.files_are_processed:
            self.process_files()
        return len(self.df)
    
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index ...
            # If data arrays are available just slice them
            if self.get_data_at_first_call or self.data_is_processed:
                x = self.images[key, ...]
                y = self.labels[key]
            # Otherwise, check to see if the key has been accessed before, and is unavailable
            elif self.df.loc[key,'is_available'] == 0:
                # Look for the first one that is available, if there is any. If not, keep looking until you find one.
                try:
                    first_available = self.df.loc[self.df['is_available']==1].index[0]
                    x = self.get_single_image(first_available, expand_dims=False)
                    y = self.df.loc[first_available,'name_idx']
                except:
                    for key_ in range(len(self.df)):
                        x = self.get_single_image(key_, expand_dims=False)
                        if x is not None:
                            y = self.df.loc[key_,'name_idx']
                            break
            # Otherwise, see if the key has been accessed before and is available
            elif self.df.loc[key,'is_available'] == 1:
                x = self.get_single_image(key, expand_dims=False)
                y = self.df.loc[key,'name_idx']
            # Otherwise, the key has not been accessed before, so we have to access it for the first time.
            else:
                image = self.get_single_image(key, expand_dims=False)
                if image is None:
                    # Look for the first one that is available, if any. If not, keep looking until you find one.
                    try:
                        first_available = self.df.loc[self.df['is_available']==1].index[0]
                        x = self.get_single_image(first_available, expand_dims=False)
                        y = self.df.loc[first_available,'name_idx']
                    except:
                        for key_ in range(len(self.df)):
                            x = self.get_single_image(key_, expand_dims=False)
                            if x is not None:
                                y = self.df.loc[key_,'name_idx']
                                break
                else:
                    x = image
                    y = self.df.loc[key,'name_idx']
                    
            x_ = torch.from_numpy(x).to(self.DTYPE).permute(2,0,1) # to make it C x H x W
            y_ = torch.tensor(y, dtype=torch.long)
            if self.transform is not None: x_ = self.transform(x_)
            return (x_, y_)
        else:
            raise TypeError("Invalid argument type. It should be an index or a slice.")
        
    
    
    def process_files(self, verbose=False):
        # Process data files
        if verbose: print("Processing files and extracting information ...")
        self.process_file(self.actors_file, sex='M', train_size=self.train_size)
        self.process_file(self.actresses_file, sex='F', train_size=self.train_size)
        self.files_are_processed = True
        
        
    def process_file(self, filename:str, sex:str='M', train_size:int=70):
        assert sex=='M' or sex=='F', "`sex` must be either 'M' or 'F'"
        idx = self.df.shape[0]
        assert filename not in self.processed_files, "This file has already been processed."
        with open(filename, 'r') as f:
            for line in f:
                artist,unk1,unk2,url,box_str,hash = line.split("\t")
                artist = artist.strip().lower().replace(" ", "_")
                if artist in self.artists_list:
                    self.artists_counts[artist] += 1
                else:
                    self.artists_list.append(artist)
                    self.artists_counts[artist] = 1
                    self._num_artists += 1
                subset = 'train' if self.artists_counts[artist] <= train_size else 'test'
                index = self.artists_list.index(artist)
                x1,y1,x2,y2 = box_str.split(',')
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                unk1 = int(unk1)
                unk2 = int(unk2)
                self.df.loc[idx] = [artist, index, sex, unk1, unk2, url, x1, x2, y1, y2, hash, subset, 2]
                if subset == 'train':
                    self.train_indices.append(idx)
                else:
                    self.test_indices.append(idx)
                idx += 1
        self.processed_files.append(filename)
        for int_col in ['name_idx','var1','var2','x1','x2','y1','y2','is_available']:
            self.df[int_col] = self.df[int_col].astype(np.int32)
        
    
    
    def get_cropped_image(self, key:int, verbose:bool=False):
        # print("------ Getting cropped image for key: {} ------ ".format(key), end='')
        if not self.files_are_processed:
            if verbose: print("Files are not processed yet. Doing so now ...")
            self.process_files(verbose=verbose)
        row = self.df.loc[key]
        url = row['url']
        name = row['name']
        extension = url.split('.')[-1]
        filename = os.path.join(self.target_folder, name + "_" + str(key) + "." + extension)
        # image = timeout(get_image_from_url, args=(url, filename), 
        #                 kwargs={'delete_file':self.delete_image_after_download, 'verbose':verbose}, 
        #                 timeout_duration=10, default=None)
        try:
            image = get_image_from_url(url, filename, delete_file=self.delete_image_after_download, 
                                       verbose=verbose)
            # print("Image DTYPE immediately after downloading: {}".format(image.dtype))
        except Exception as e:
            if verbose:
                print("Error in get_image_from_url(%s):"%url)
                print(e)
            image = None
            
        if image is None:
            if verbose: 
                print(("\nWARNING in get_cropped_image(%d): "%key) + 
                ("We could not get any readable image from URL %s"%url))
            self.df.loc[key, 'is_available'] = 0
            # print("FAILED")
            # display(self.df['is_available'].value_counts())
            # print("Count of NAN values in dataframe: {}".format(self.df['is_available'].isna().sum()))
            return None
        else:
            if len(image.shape) == 3:
                # print("SUCCESS")
                # self.df.loc[key, 'is_available'] = 1 # Don't hurry up yet. We still need to properly resize the image.
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                # display(self.df['is_available'].value_counts())
                # print("Count of NAN values in dataframe: {}".format(self.df['is_available'].isna().sum()))
                return image[y1:y2, x1:x2, :]
            else:
                if verbose: 
                    print(("\nWARNING in get_cropped_image(%d): "%key) + 
                    ("The image from URL %s is not RGB;"%url))
                    print("It has shape %s"%str(image.shape))
                self.df.loc[key, 'is_available'] = 0
                # print("FAILED")
                # print(self.df['is_available'].value_counts())
                # print("Count of NAN values in dataframe: {}".format(self.df['is_available'].isna().sum()))
                return None
        
    
    
    def resize(self, image_arr, shape=None, verbose=False):
        if image_arr is None:
            return None
        else:
            if shape is None:
                shape = self.target_image_size
            try:
                resized_image = np.round(
                    skimage.transform.resize(image_arr, (shape, shape), preserve_range=True)).astype(np.uint8)
                # print("Image DTYPE after resizing: ", resized_image.dtype)
                return resized_image
            except Exception as e:
                if verbose:
                    print("Error in resize():")
                    print(e)
                return None
            
    
    
        
    def get_single_image(self, key:int, expand_dims:bool=False, verbose:bool=False, check_for_existence:bool=True):
        try:
            if self.data_is_processed:
                try:
                    image = self.images[key]
                except Exception as e:
                    if verbose:
                        print("Could not access image from self.images. Returning none. Error is: ")
                        print(e)
                    image = None
            else:
                if self.df.loc[key,'is_available'] > 0:
                    image = self.resize(self.get_cropped_image(key, verbose=verbose), verbose=verbose)
                else:
                    image = None
                    
            if image is None:
                self.df.loc[key,'is_available'] = 0
                return None
            else:
                if list(image.shape) == [self.target_image_size, self.target_image_size, 3]:
                    self.df.loc[key,'is_available'] = 1
                    if expand_dims:
                        return np.expand_dims(image, axis=0)
                    else:
                        return image
                else:
                    if verbose:
                        print("WARNING in get_single_image(%d): "%(key))
                        print("The image has shape %s"%(str(image.shape)))
                        print("Returning None.")
                    self.df.loc[key,'is_available'] = 0
                    return None
        except Exception as e:
            if verbose:
                print("Error in get_single_image(%d):"%key)
                print(e)
            self.df.loc[key,'is_available'] = 0
            return None
    
    
    
    def get_all_images(self, return_images:bool=False, return_labels:bool=False, verbose:bool=False):
        begin = len(self.images)
        for key in tqdm(range(begin, len(self.df)), desc='Progress', ncols=100):
            image = self.get_single_image(key, expand_dims=True, verbose=verbose)
            if image is not None:
                self.images.append(image)
                self.labels.append(self.df.loc[key, 'name_idx'])
        try:
            self.images = np.concatenate(self.images, axis=0)
            self.labels = np.array(self.labels, dtype=np.int32)
        except Exception as e:
            if verbose:
                print("Error when concatenating in get_all_images():")
                print(e)
                print("Unconcatenated ararys will be returned if requested.")
        if return_images:
            if return_labels:
                return self.images, self.labels
            else:
                return self.images
    
    
    
    def process_all_data(self, verbose:bool=False):
        if verbose:
            print("Processing all data ...")
            print("Current number of records in the data frame: {}".format(self.df.shape[0]))
        self.get_all_images(verbose=verbose)
        self.df = self.df.loc[self.df['is_available']==1]
        self.df.reset_index(drop=True, inplace=True)
        if verbose: print("Recalculating training and testing indices from shrunk dataset ...")
        self.train_indices = self.df.loc[self.df['subset']=='train'].index.tolist()
        self.test_indices = self.df.loc[self.df['subset']=='test'].index.tolist()
        if verbose: 
            print("New number of records in the data frame: {}".format(self.df.shape[0]))
        assert len(self.df) == len(self.images), \
            "Something went wrong with the data processing. Shapes mismatch. "+\
            "Dataframe: {}, Images: {}".format(len(self.df), len(self.images))
        self.data_is_processed = True
        
    
          
    def save_data(self, path:str, verbose:bool=False):
        if "." not in path or \
            (".csv" not in path and ".txt" not in path and ".pkl" not in path and ".h5" not in path and \
            ".parquet" not in path and ".feather" not in path and ".pickle" not in path):
            mode = "folder"
            if verbose: print("Saving all data in a directory ...")
        else:
            mode = "file"
            if verbose: print("Saving all data in a file ...")
        
        if not self.data_is_processed:
            self.process_all_data(verbose=verbose)
        
        if mode == "folder":
            df = self.df.iloc[:,:13]
            images = self.images
            labels = self.labels
            artists = np.array(self.artists_list, dtype=str)
            if verbose:
                print("Shape of images array:  {}".format(images.shape))
                print("Shape of labels array:  {}".format(labels.shape))
                print("Shape of artists array: {}".format(artists.shape))
                print("Shape of dataframe:     {}".format(df.shape))
            np.savez(make_path(os.path.join(path,"data.npz")), images=images, labels=labels, artists=artists)
            try:
                df.to_parquet(make_path(os.path.join(path, "info.parquet")))
            except:
                df.to_csv(make_path(os.path.join(path, "info.csv")), index=False, header=True)
        else:    
            self.image_cols = ['i_%d'%(i) for i in np.arange(self.images[0,:].size).tolist()]
            if verbose: 
                print("Shape of data frame BEFORE augmenting it: {}".format(self.df.shape))
                print("Shape of images array BEFORE augmenting dataframe: {}".format(self.images.shape))
                # print("Dataframe BEFORE augmentation: ")
                # print(self.df)
                print("Augmenting the dataframe ...")
            try:
                df_aug = pd.DataFrame(
                    np.zeros((self.df.shape[0],len(self.image_cols)), dtype=np.uint8), 
                    columns=self.image_cols, dtype=np.uint8)
                self.df = self.df.join(df_aug)
            except Exception as e:
                print("Error while augmenting the dataframe: ")
                print(e)
                raise e
            for key in range(len(self.df)):
                # print("Working on key: {}".format(key))
                img = self.images[key,...].flatten()
                # print("\tShape of image: {}".format(img.shape))
                self.df.loc[key,self.image_cols] = img
                # print("\tdataframe at this time: ")
                # print(self.df)
                # print("\tShape of dataframe at this time: {}".format(self.df.shape))
            if verbose: 
                print("Shape of data frame AFTER augmenting it: {}".format(self.df.shape))
                print("Shape of images array AFTER augmenting dataframe: {}".format(self.images.shape))
            filename = path
            for col in self.image_cols:
                self.df[col] = self.df[col].astype(np.uint8)
            if '.csv' in filename or '.txt' in filename:
                self.df.to_csv(filename, header=True, index=False)
            elif '.pkl' in filename or '.pickle' in filename:
                self.df.to_pickle(filename)
            elif '.h5' in filename:
                self.df.to_hdf(filename, key='df', mode='w')
            elif '.feather' in filename:
                self.df.to_feather(filename)
            elif '.parquet' in filename:
                self.df.to_parquet(filename)
    
    
    
    def load_data(self, path:str, verbose:bool=False):
        
        mode = "file"
        filename = path
        if '.csv' in filename or '.txt' in filename:
            self.df = pd.read_csv(filename)
        elif '.pkl' in filename or '.pickle' in filename:
            self.df = pd.read_pickle(filename)
        elif '.h5' in filename:
            self.df = pd.read_hdf(filename)
        elif '.feather' in filename:
            self.df = pd.read_feather(filename)
        elif '.parquet' in filename:
            self.df = pd.read_parquet(filename)
        elif os.path.isdir(filename):
            mode = "folder"
        else: # Tehn what the hell is the path?
            raise ValueError("The path is not a valid file or directory.")
        
        if mode == "file":    
            if len(self.df.columns) <= 13:
                if verbose: print("WARNING: The loaded data does not contain the images. No new images will be loaded.")
            else:
                self.image_cols = list(self.df.columns[13:])
                images = self.df[self.image_cols].values
                self.labels = self.df['name_idx'].values
                self.images = []
                for key in range(len(self.df)):
                    img = images[key,...]
                    img_reshaped = np.reshape(img, (self.target_image_size, self.target_image_size, 3))
                    img_expanded = np.expand_dims(img_reshaped, 0)
                    self.images.append(img_expanded)
                self.images = np.concatenate(self.images, axis=0)
        else:
            npzfile = np.load(os.path.join(filename, "data.npz"))
            self.images = npzfile['images']
            self.labels = npzfile['labels']
            self.artists_list = npzfile['artists'].tolist()
            try:
                self.df = pd.read_parquet(os.path.join(filename, "info.parquet"))
            except:
                self.df = pd.read_csv(os.path.join(filename, "info.csv"))
        
        assert len(self.df) == len(self.images), \
            "Number of images loaded is not equal to the number of rows in the dataframe. Something went wrong."
        
        self.train_indices = self.df.loc[self.df['subset']=='train'].index.tolist()
        self.test_indices = self.df.loc[self.df['subset']=='test'].index.tolist()
        self.data_is_processed = True
        self.files_are_processed = True
        
            
        
            
        
        


        
def get_image_from_url(url, filename, delete_file=True, return_response:bool=False, verbose:bool=False):
    try:
        response = urllib.request.urlretrieve(url,filename)
    except Exception as e:
        if verbose:
            print("\nError in urllib.request.urlretrieve('%s'): "%url)
            print(e)
        raise e
    try:
        image = skimage.io.imread(filename)
        # print(image.dtype)
    except Exception as e:
        if verbose:
            print("\nError in skimage.io.imread('%s'): "%filename)
            print(e)
            print("Deleting the faulty file ...")
        try:
            os.remove(filename)
        except Exception as e_:
            if verbose:
                print("\nError in os.remove('%s'): "%filename)
                print(e_)
            raise e_
        else:
            raise e
    if delete_file==True: 
        try:
            os.remove(filename)
        except Exception as e:
            if verbose:
                print("\nError in os.remove('%s'): "%filename)
                print(e)
            raise e
    if return_response:
        return image, response
    else:
        return image

            
    
    

def make_train_test_datasets(actors_file:str, actresses_file:str, target_image_size:int=28, 
                             target_folder:str=None, transform=None, train_size:int=70, 
                             get_data_at_first_call:bool=False, verbose:bool=False):
    if verbose: print("\nConstructing master dataset ...")
    master_dataset = FaceScrubDataset(actors_file, actresses_file, target_image_size, 
                               target_folder, transform, train_size, get_data_at_first_call, verbose=verbose)
    if verbose: print("\nConstructing training and testing datasets ...")
    training_dataset = Subset(master_dataset, master_dataset.train_indices)
    test_dataset = Subset(master_dataset, master_dataset.test_indices)
    return master_dataset, training_dataset, test_dataset


def split_dataset(dataset):
    training_dataset = Subset(dataset, dataset.train_indices)
    test_dataset = Subset(dataset, dataset.test_indices)
    return training_dataset, test_dataset
    



def show_images_from_datasets(masterset,trainset,testset,nrows,ncols,verbose=True,num_workers=0,savefig=None):
    if verbose: print("\nCreating DataLoader for master dataset...")
    t1 = time.time()
    masterloader = DataLoader(masterset, batch_size=nrows*ncols, shuffle=True, num_workers=num_workers)
    t2 = time.time()
    if verbose: print("Time elapsed for creating data loader with batch size of %d: %.2f seconds."%(nrows*ncols,t2-t1))
    if verbose: print("\nShowing images from master dataset ...")
    t1 = time.time()
    test_images_from_data_loader(masterloader,nrows,ncols,suptitle="Master Dataset",savefig=savefig,verbose=verbose)
    t2 = time.time()
    if verbose: print("Time elapsed for showing %d images: %.2f seconds"%(nrows*ncols,t2-t1))
    if verbose: print("\nCreating DataLoader for training dataset...")
    t1 = time.time()
    trainloader = DataLoader(trainset, batch_size=nrows*ncols, shuffle=True, num_workers=num_workers)
    t2 = time.time()
    if verbose: print("Time elapsed for creating data loader with batch size of %d: %.2f seconds."%(nrows*ncols,t2-t1))
    if verbose: print("\nShowing images from training dataset ...")
    t1 = time.time()
    test_images_from_data_loader(trainloader,nrows,ncols,suptitle="Training Dataset",savefig=savefig,verbose=verbose)
    t2 = time.time()
    if verbose: print("Time elapsed for showing %d images: %.2f seconds"%(nrows*ncols,t2-t1))
    if verbose: print("\nCreating DataLoader for testing dataset...")
    t1 = time.time()
    testloader = DataLoader(testset, batch_size=nrows*ncols, shuffle=True, num_workers=num_workers)
    t2 = time.time()
    if verbose: print("Time elapsed for creating data loader with batch size of %d: %.2f seconds."%(nrows*ncols,t2-t1))
    if verbose: print("\nShowing images from testing dataset ...")
    t1 = time.time()
    test_images_from_data_loader(testloader,nrows,ncols,suptitle="Testing Dataset",savefig=savefig,verbose=verbose)
    t2 = time.time()
    if verbose: print("Time elapsed for showing %d images: %.2f seconds"%(nrows*ncols,t2-t1))
    
    
        
        
def test_images_from_data_loader(dataloader, nrows, ncols, suptitle=None, savefig=None, verbose=False):
    plt.figure(figsize=(ncols*3,nrows*3))
    if suptitle: plt.suptitle(suptitle)
    batch = next(iter(dataloader))
    (X,Y) = batch
    if verbose:
        print("Shape of input batch: ", X.shape)
        print("Shape of label batch: ", Y.shape)
    for i in range(nrows*ncols):
        x = X[i]
        y = Y[i]
        if i == 0 and verbose:
            print("Shape of input data point: ", x.shape)
            print("Shape of label data point: ", y.shape)
        plt.subplot(nrows,ncols,i+1)
        plt.imshow(x.permute(1,2,0), vmin=0, vmax=255)
        try:
            plt.title(dataloader.dataset.artists_list[y])
        except:
            plt.title(dataloader.dataset.dataset.artists_list[y])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #plt.show()
    if savefig:
        plt.savefig(("%s_%s.png"%(savefig,suptitle)), dpi=330)   
    






if __name__ == '__main__':
    
    print("yep.")
    # cwd = os.getcwd()
    # print("Current working directory: {}".format(cwd))
    # actors_file = "./assignment2/data/facescrub/new_subset_actors.txt"
    # actresses_file = "./assignment2/data/facescrub/new_subset_actresses.txt"
    # # num_lines = 5
    # # new_actors_file = "./assignment2/data/facescrub/new_subset_actors_%d.txt"%num_lines
    # # new_actresses_file = "./assignment2/data/facescrub/new_subset_actresses_%d.txt"%num_lines
    # # f_m = open(new_actors_file, 'w')
    # # f_f = open(new_actresses_file, 'w')
    # # with open(actors_file, 'r') as f:
    # #     for _ in range(num_lines):
    # #         line = f.readline()
    # #         f_m.write(line)
    # # with open(actresses_file, 'r') as f:
    # #     for _ in range(num_lines):
    # #         line = f.readline()
    # #         f_f.write(line)
    # # f_m.close()
    # # f_f.close()
    # # print("Creating new dataset files complete.")
    # print("\nCreating datasets without getting all the images at the beginning ...")
    # master_dataset, train_dataset, test_dataset = \
    #     make_train_test_datasets(actors_file, actresses_file, target_image_size=28, 
    #                          target_folder="./assignment2/data/facescrub/uncropped/", transform=None, train_size=70, 
    #                          get_data_at_first_call=False)
    # print("Datasets have been created.")
    # print("Number of training images: {}".format(len(train_dataset)))
    # print("Number of test images: {}".format(len(test_dataset)))
    # # print("\nTrying to access and show a few of the images ...\n")
    # # show_images_from_datasets(master_dataset,train_dataset,test_dataset,5,5,verbose=True,
    # #                           savefig="assignment2/data/facescrub/test", num_workers=0)
    # # print("\nChecking to see how many image indices could not be downloaded from each dataset ...")
    # # print("Checking master dataset ...")
    # # display(master_dataset.df['is_available'].value_counts())
    # # display(master_dataset.df)
    # # print("Trying to get all the images of the master dataset...")
    # # t1 = time.time()
    # # master_dataset.get_all_images(return_array=False, verbose=False)
    # # t2 = time.time()
    # # print("Time elapsed: %.2f seconds."%(t2-t1))
    # # print("\nChecking to see how many image indices could not be downloaded from each dataset ...")
    # # display(master_dataset.df['is_available'].value_counts())
    # # master_dataset.df.to_csv("./assignment2/data/facescrub/data_raw_%d.csv"%num_lines)
    # print("\nTrying to save all the images of the training dataset...")
    # master_dataset.save_data("./assignment2/data/facescrub/data_size_28_training_70.parquet")
    # print("Done writing to file.")
    # print("Checking the shape of the data frame at the end ...")
    # print(master_dataset.df.shape)
    # print("All done. Good bye!")
    # # print("Checking data types of the data frame at the end ...")
    # # print(master_dataset.df.dtypes)
    
                                                                            
    
    
