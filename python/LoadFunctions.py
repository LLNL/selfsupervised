import cv2
import numpy as np
import copy
import sys
import signal

def ctrl_c_handler(signum, frame):
    # do nothing, return and absorb the ctrl-c
    # otherwise, we go bananas when we try to exit. 
    pass 


# *******************************************************************************************************************
def get_resize_flag():
    """This function when called will randomly return one of four resize flags for OpenCV to use. 
    """
    resize_flag = np.random.randint(0,4)
    
    if resize_flag == 0:
        flag   = cv2.INTER_LINEAR
    elif resize_flag == 1:
        flag   = cv2.INTER_AREA
    elif resize_flag == 2:
        flag   = cv2.INTER_CUBIC
    elif resize_flag == 3:
        flag   = cv2.INTER_LANCZOS4   
        
    return flag

# *******************************************************************************************************************
def flip_and_rotate_w_classification(in_class, img1, img2, img3):
    """We will will mirror flip patches and then do rotation with classification (RWC).
    
    The odds of a left-right mirror flip are 50%. If we do a left-right flip, we have to relabel the class to match the 
    new configuration. So, a good deal of this function body is dedicated to relabeling a class on left-right mirror flipping. 
    
    When we do rotation with classification, we increment the class in steps of 20. 
    
    This function returns the updated class and the new flipped and/or rotated patches. 
    """
    
    # roll a 0 or 1
    flip = np.random.randint(0,2)
     
    assert(flip < 2)
    assert(flip >= 0)
    
    in_class_val = int(in_class)
    
    # To flip or not to flip
    if flip == 0:
        nimg1       = img1
        nimg2       = img2 
        nimg3       = img3
        out_class   = in_class_val
    else:
        # Notes: 
        # (1) some classes return the same since they are symmetric
        # (2) The 4 patches have to be swapped
        
        if in_class_val > 11:

            inv_flip = False
            
            nimg1 = np.fliplr(img1)
            
            if in_class_val == 12:
                out_class = 14           
            elif in_class_val == 14:
                out_class = 12
            elif in_class_val == 17:
                out_class = 19         
            elif in_class_val == 19:
                out_class = 17
            elif in_class_val == 13:
                out_class = 15
                inv_flip  = True
            elif in_class_val == 15:
                out_class = 13
                inv_flip  = True
            elif in_class_val == 16:
                out_class = 18
                inv_flip  = True            
            elif in_class_val == 18:
                out_class = 16
                inv_flip  = True    
             
            if inv_flip:
                nimg2 = np.fliplr(img3) 
                nimg3 = np.fliplr(img2) 
            else:
                nimg2 = np.fliplr(img2) 
                nimg3 = np.fliplr(img3)
                
        elif in_class_val > 7:
            nimg1 = np.fliplr(img3)
            nimg2 = np.fliplr(img2) 
            nimg3 = np.fliplr(img1)
            if in_class_val == 8:
                out_class = 11
            elif in_class_val == 11:
                out_class = 8
            elif in_class_val == 9:  
                out_class = 10
            elif in_class_val == 10:  
                out_class = 9  
        else:
            nimg1 = np.fliplr(img1)
            nimg2 = np.fliplr(img2) 
            nimg3 = np.fliplr(img3)
            
            if in_class_val == 0:
                out_class = 2
            elif in_class_val == 2:
                out_class = 0
            elif in_class_val == 3:
                out_class = 7
            elif in_class_val == 7:
                out_class = 3  
            elif in_class_val == 4: 
                out_class = 6
            elif in_class_val == 6: 
                out_class = 4
            elif in_class_val == 1: 
                out_class = 1
            elif in_class_val == 5: 
                out_class = 5
    
    # 80 Class rotation with classification
    
    # Roll a 0,1,2 or 3
    flip = np.random.randint(0,4)
     
    if flip == 1:   # Rotate 90 Degrees
        nimg1 = np.rot90(nimg1)
        nimg2 = np.rot90(nimg2) 
        nimg3 = np.rot90(nimg3)   
        out_class += 20 
    elif flip == 2:  # Rotate 180 Degrees       
        nimg1 = np.rot90(nimg1,k=2)
        nimg2 = np.rot90(nimg2,k=2) 
        nimg3 = np.rot90(nimg3,k=2)    
        out_class += 40      
    elif flip == 3:  # Rotate 270 Degrees 
        nimg1 = np.rot90(nimg1,k=3)
        nimg2 = np.rot90(nimg2,k=3) 
        nimg3 = np.rot90(nimg3,k=3)    
        out_class += 60  
             
    return out_class, nimg1, nimg2, nimg3

# *******************************************************************************************************************
def load_cv_image_triple(proc, img_name_1, img_name_2, img_name_3, in_class, min_size, max_size, patch_size, rand_seed, mean_color):
    """This function will launch in its own process from a pool to load and augment image patches.
    
    We use OpenCV to load in three patches. We then decide how much to resize the patches, and then we do that.
    We then decide on the size of the internal aperture and apply it randomly to two of the patches. 
    
    We return the process ID so we can do some safety book keeping. We also return the three patches and the 
    updated class from executing flipping and rotation with classification. 
    """
    try:
        # Put in a dummy ctrl-c handler so this process will ignore it
        signal.signal(signal.SIGINT, ctrl_c_handler)
        
        assert(patch_size   >= min_size)
        assert(max_size     >= min_size)
        assert(patch_size   <= max_size)
        
        do_exit = False
        
        # set our random number seed
        np.random.seed(rand_seed)
        
        # load in images
        img_1 = cv2.imread(img_name_1)
        img_2 = cv2.imread(img_name_2)
        img_3 = cv2.imread(img_name_3)
        
        # Check images to make sure they loaded
        if type(img_1) is not np.ndarray:
            print (">>>>> Bad Image {}".format(img_name_1)) 
            do_exit = True   
            
        if type(img_2) is not np.ndarray:
            print (">>>>> Bad Image {}".format(img_name_2))
            do_exit = True
        
        if type(img_3) is not np.ndarray:
            print (">>>>> Bad Image {}".format(img_name_3))
            do_exit = True
        
        # determine patch size and how to rescale it    
        y_size      = float(img_1.shape[0])
        x_size      = float(img_1.shape[1])
    
        new_size = np.random.randint(min_size,max_size+1)
    
        if y_size > x_size:
            new_y_size  = new_size 
            new_x_size  = int(round(new_size * (x_size/y_size)))
        elif y_size < x_size:
            new_x_size  = new_size 
            new_y_size  = int(round(new_size * (y_size/x_size)))
        else:
            new_x_size  = new_size 
            new_y_size  = new_size 
    
        # Apply yoked random jitter on patches while we resize the patches
        new_pos_x       = np.random.randint(0,new_x_size-patch_size+1)
        new_pos_y       = np.random.randint(0,new_y_size-patch_size+1)
    
        new_img_1       = cv2.resize(img_1, (new_x_size,new_y_size))[new_pos_y:new_pos_y+patch_size,new_pos_x:new_pos_x+patch_size,:]
        
        new_img_2       = cv2.resize(img_2, (new_x_size,new_y_size))[new_pos_y:new_pos_y+patch_size,new_pos_x:new_pos_x+patch_size,:]
        
        new_img_3       = cv2.resize(img_3, (new_x_size,new_y_size))[new_pos_y:new_pos_y+patch_size,new_pos_x:new_pos_x+patch_size,:]
     
        # Determine the size and position of the random aperture 
        # Also, add 1 since randint is exlusive with the last number in range
        internal_size           = np.random.randint(patch_size-32,patch_size+1) 
        
        internal_pos_x          = np.random.randint(0,patch_size-internal_size+1)
        internal_pos_y          = np.random.randint(0,patch_size-internal_size+1)
        
        # Which patch does not get apertured?
        leave_ok_roll           = np.random.randint(0,3)
        
        # create a new empty image template and set it to mean gray
        return_img_tmpl         = np.empty((patch_size, patch_size, 3),dtype=np.uint8)
                
        return_img_tmpl[:,:,0]  = mean_color[0]
        return_img_tmpl[:,:,1]  = mean_color[1]
        return_img_tmpl[:,:,2]  = mean_color[2]
        
        # set pointers to the images we will apply aperture to
        if leave_ok_roll == 0:
            new_img_A = new_img_1
            new_img_B = new_img_2
        elif leave_ok_roll == 1:
            new_img_A = new_img_1
            new_img_B = new_img_3
        else:
            new_img_A = new_img_2
            new_img_B = new_img_3
        
        # make one more copy of mean gray image
        return_img_A = return_img_tmpl
        return_img_B = copy.deepcopy(return_img_tmpl)
        
        # copy the visible area over the mean gray image to create the aperture images
        return_img_A[internal_pos_y:internal_pos_y+internal_size, internal_pos_x:internal_pos_x+internal_size, :] = \
            new_img_A[internal_pos_y:internal_pos_y+internal_size, internal_pos_x:internal_pos_x+internal_size, :]
        return_img_B[internal_pos_y:internal_pos_y+internal_size, internal_pos_x:internal_pos_x+internal_size, :] = \
            new_img_B[internal_pos_y:internal_pos_y+internal_size, internal_pos_x:internal_pos_x+internal_size, :]
        
        # return the aperture image patches in the correct order. 
        if leave_ok_roll == 0:        
            out_class, nimg1, nimg2, nimg3 = flip_and_rotate_w_classification(in_class, return_img_A,  return_img_B,   new_img_3)
        elif leave_ok_roll == 1:
            out_class, nimg1, nimg2, nimg3 = flip_and_rotate_w_classification(in_class, return_img_A,  new_img_2,      return_img_B)
        else:
            out_class, nimg1, nimg2, nimg3 = flip_and_rotate_w_classification(in_class, new_img_1,     return_img_A,   return_img_B)
            
    except Exception as e:
        print('Error LoadFunctions on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
        do_exit = True
        
    return proc, nimg1, nimg2, nimg3, out_class, do_exit

# *******************************************************************************************************************
# *******************************************************************************************************************
# EXPERIMENTAL
# *******************************************************************************************************************
# *******************************************************************************************************************         


