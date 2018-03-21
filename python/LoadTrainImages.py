import caffe
import numpy as np
import random
import copy
from multiprocessing import Pool, TimeoutError
import sys
import LoadFunctions as loadf
       
# *******************************************************************************************************************
class Load:
    """ This is a loader class. We will create a few instance in different threads so we have a rotating buffer.
    
    The class has a variety of parameters that need to be set once. Otherwise, we just call load_triple_batch each
    time we want to load a bunch of patches. The class will take care of shuffling the order of samples as well as
    any processing we need to apply to each patch. Loading patches is handled via process pool since Python threading
    is weird. Take a look at PrefetchTrain.py for how this is initialized and called.  
    """    
    def __init__(self, batch_size, patch_size, solver, mean_val, image_file):
        """This initialized the loader class. See: train_batch_pre_load_triple for example of how this works
        """
        # Initial default values
        self._use_triple                = False
        self._use_load_pool             = False
        self._use_ra                    = False
                
        self._mean_color                = np.float32(mean_val)
        
        # Simple variables that we need to set local as a copy
        self._batch_size                = copy.deepcopy(batch_size)
        self._patch_size                = copy.deepcopy(patch_size)
        
        # complex objects and lists
        self._image_file                = image_file
        self._path_prefix               = ""
        
        # Allocate some memories we will use several times that hold the loaded patches
        self._image_set_1               = np.zeros((batch_size,3,patch_size,patch_size))
        self._image_set_2               = np.zeros((batch_size,3,patch_size,patch_size))
        self._image_set_3               = np.zeros((batch_size,3,patch_size,patch_size))
        self._label_set                 = batch_size*[None] 
        
        self._load_pool                 = None
        self._img_idx                   = []
        
        # Set up default random aperture parameters, we will override these later
        self._ra_min_size               = 0
        self._ra_max_size               = 0
        self._ra_patch_size             = 0
                
        # go ahead and do an initial shuffle of the data    
        self.shuffleIdx()
                
    def useRandomAperture(self, min_size, max_size, patch_size):
        """Set up random aperture parameters
        """
        self._use_ra                   = True
        self._ra_min_size              = min_size
        self._ra_max_size              = max_size
        self._ra_patch_size            = patch_size
                
    def useTriple(self, solver):        
        """Set up triple patch parameters and transformers
        """
        self._use_triple                     = True 

        self._train_transformer_1 = copy.deepcopy(caffe.io.Transformer({'data_1': solver.net.blobs['data_1'].data.shape}))
        self._train_transformer_1.set_transpose('data_1', (2,0,1))
        self._train_transformer_1.set_mean('data_1', self._mean_color) # mean pixel
            
        self._train_transformer_2 = copy.deepcopy(caffe.io.Transformer({'data_2': solver.net.blobs['data_2'].data.shape}))
        self._train_transformer_2.set_transpose('data_2', (2,0,1))
        self._train_transformer_2.set_mean('data_2', self._mean_color) # mean pixel  
        
        self._train_transformer_3 = copy.deepcopy(caffe.io.Transformer({'data_3': solver.net.blobs['data_3'].data.shape}))
        self._train_transformer_3.set_transpose('data_3', (2,0,1))
        self._train_transformer_3.set_mean('data_3', self._mean_color) # mean pixel
             
    def useLoadPool(self, procs = 0):
        """Set up the loading pool processes. 
        """
        self._use_load_pool  = True
        if procs == 0:
            self._load_pool      = Pool(processes=self._batch_size) # probably a bad idea
        else:
            self._load_pool      = Pool(processes=procs) 
                
    def shuffleIdx(self):
        """Shuffle the data by shuffling an index which is much faster and easier than shuffling the actual data.
        """
        
        print ("Shuffle data")
        
        self._img_idx = []
    
        list_length = self._image_file.length()

        print ("    {} items".format(list_length))
    
        for x in range(list_length ):
            self._img_idx.append(x)
        
        random.shuffle(self._img_idx)
        
        print ("... Done")    
        
    def loadTripleBatch(self):
        """Loader for the triple patches which run in their own thread and call a pool of loader processes. 
        
        This function will get the shuffled list of patches along with their labels. These are then fed to the loading
        pool of processes. This will then get batch_size number of patch sets. Note that if we are using multiple GPU's 
        the local batch size is divided by the total number of GPU's requested.
        """
        try:
            
            assert(self._use_ra         == True)
            assert(self._use_triple     == True)
            assert(self._use_load_pool  == True)
            assert(self._image_file.__class__.__name__ == "CompactList")
            
            do_exit = False
            
            #if we have fewer indices than the batch size, do a new shuffle
            if len(self._img_idx) < self._batch_size:
                self.shuffleIdx()
            
            my_cv_img1  = []
            my_cv_img2  = []
            my_cv_img3  = []
            my_label    = []
            
            my_im_name1 = []
            my_im_name2 = []
            my_im_name3 = []
            
            # for each patch set in the local batch size, get the file name and label
            for i in range(self._batch_size):    
                
                # if we run out of indices, do a new shuffle
                if len(self._img_idx) == 0:
                    self.shuffleIdx()
                
                # get the new index
                idx = self._img_idx.pop()    
                
                # get the file name
                files, ilabel   = self._image_file.getFileNames(idx)
                im_name1        = self._path_prefix + files[0] 
                im_name2        = self._path_prefix + files[1]    
                im_name3        = self._path_prefix + files[2]              
  
                # append file names and class labels to our list
                my_im_name1.append(im_name1)
                my_im_name2.append(im_name2)
                my_im_name3.append(im_name3)   
                my_label.append(ilabel)
  
            assert(self._use_load_pool == True)
            
            # roll a different random number seed for each loading process    
            rand_seed = []
            for i in range(self._batch_size):
                r = np.random.randint(0,pow(2,32)) 
                rand_seed.append(r)
            
            # Launch the load pool processes and load/process the patch images. 
            multiple_results = [self._load_pool.apply_async(loadf.load_cv_image_triple,
                                                            (i, my_im_name1[i], my_im_name2[i], my_im_name3[i], my_label[i],
                                                            self._ra_min_size, self._ra_max_size, self._ra_patch_size, rand_seed[i], self._mean_color))
                                for i in range(self._batch_size)]
            
            # for each process, return the patch images and class
            # check a counter to make sure things were kept in order                                       
            try:
                vals        = [res.get() for res in multiple_results]
                counter     = 0
                for i in vals:
                    my_cv_img1.append(i[1])
                    my_cv_img2.append(i[2])
                    my_cv_img3.append(i[3])
                    self._label_set[counter] = i[4]
                    if i[5] == True: do_exit = True
                    assert(i[0] == counter)     # Make sure order was preserved by process pool
                    counter += 1
                    
            except TimeoutError:
                print "multiprocessing.TimeoutError"
      
            # For each patch set, apply Caffe's preprocessor and check sizing to be paranoid. 
            for i in range(self._batch_size): 
                
                nimg1 = my_cv_img1[i] 
                nimg2 = my_cv_img2[i] 
                nimg3 = my_cv_img3[i]
                  
                in_img                                      = self._train_transformer_1.preprocess('data_1', nimg1) 
                if in_img.shape[2] is not self._patch_size:
                    print("1 Patch wrong sized!!!") 
                    print("Should be {} got {}".format(self._patch_size,in_img.shape[2]))
                    do_exit = True
                if in_img.shape[1] is not self._patch_size:
                    print("2 Patch wrong sized!!!")
                    print("Should be {} got {}".format(self._patch_size,in_img.shape[1]))
                    do_exit = True
                self._image_set_1[i,:,:,:]                  = in_img
                
                in_img                                      = self._train_transformer_2.preprocess('data_2', nimg2)  
                if in_img.shape[2] is not self._patch_size:
                    print("3 Patch wrong sized!!!") 
                    print ("Should be {} got {}".format(self._patch_size,in_img.shape[2]))
                    do_exit = True
                if in_img.shape[1] is not self._patch_size:
                    print("4 Patch wrong sized!!!")
                    print("Should be {} got {}".format(self._patch_size,in_img.shape[1]))
                    do_exit = True
                self._image_set_2[i,:,:,:]                  = in_img
      
                in_img                                      = self._train_transformer_3.preprocess('data_3', nimg3)   
                if in_img.shape[2] is not self._patch_size:
                    print("5 Patch wrong sized!!!") 
                    print("Should be {} got {}".format(self._patch_size,in_img.shape[2]))
                    do_exit = True
                if in_img.shape[1] is not self._patch_size:
                    print("6 Patch wrong sized!!!")
                    print("Should be {} got {}".format(self._patch_size,in_img.shape[1]))
                    do_exit = True
                self._image_set_3[i,:,:,:]                  = in_img
    
        except Exception as e:
            print('Error in LoadTrainImages on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            do_exit = True
        # return the patch sets in the batch along with the class labels
        return self._image_set_1, self._image_set_2, self._image_set_3, self._label_set, do_exit  
