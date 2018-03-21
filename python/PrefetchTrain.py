import numpy as np
import cv2
import sys
import pylab
import types
import os
import LoadTrainImages
import matplotlib.pyplot as plt
import multiprocessing
import string
import subprocess
import shutil
from multiprocessing.pool   import ThreadPool
from caffe.proto            import caffe_pb2
from google.protobuf        import text_format

# *******************************************************************************************************************
class TrainParams:
    """Stand alone container class for parameters used in training.
    
    Typically, you will see this class object abbreviated as 'tp'.  
    """
    def __init__(self, solver, patch_size, patch_marg_1, patch_marg_2, train_batch_size, test_batch_size,
                 bin_num, image_mean, loss_layer, softmax_layer):
        
        self.solver                 = solver
        self.patch_size             = patch_size
        self.patch_marg_1           = patch_marg_1
        self.patch_marg_2           = patch_marg_2
        self.train_batch_size       = train_batch_size
        self.test_batch_size        = test_batch_size
        self.bin_num                = bin_num           # Number of class bins
        self.image_mean             = image_mean
        self.loss_layer             = loss_layer
        self.softmax_layer          = softmax_layer
        self.test_skip              = 0
        self.pool_procs             = multiprocessing.cpu_count() 
        self.test_iters             = -1
        self.ra_min_size            = patch_size
        self.ra_max_size            = patch_size
        self.ra_patch_size          = patch_size
        self.path_prefix            = ""

# *******************************************************************************************************************  
def check_file(file_name,rank = 0):
    # Check if your files are here
    if os.path.isfile(file_name):
        return 1
    else:
        if rank == 0:
            print("Error: file not found \'{}\'".format(file_name))
            print("Be sure and edit the python path to match your location")
        return 0
    
# *******************************************************************************************************************  
def check_dir(dir_name,rank = 0):
    # Check if your files are here
    if os.path.isdir(dir_name):
        return 1
    else:
        if rank == 0:
            print("Error: directory not found \'{}\'".format(dir_name))
            print("Be sure and edit the python path to match your location")
        return 0
    
# *******************************************************************************************************************  
def check_create_dir(dir_name,rank = 0):
    # Check if your files are here
    if os.path.isdir(dir_name):
        pass
    else:
        if rank == 0:
            print("Creating Directory: \'{}\'".format(dir_name))
            os.mkdir(dir_name) 
            
# ******************************************************************************************************************* 
def check_X_is_running():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0

# *******************************************************************************************************************             
def read_proto_solver_file(filepath):
    """Helper to read Caffe solver files
    """
    _ = check_file(filepath)
    solver_config = caffe_pb2.SolverParameter()
    return read_proto_file(filepath, solver_config)

# *******************************************************************************************************************
def read_proto_file(filepath, parser_object):
    """Helper to read Caffe solver files
    """
    _ = check_file(filepath)
    my_file = open(filepath, "r")

    text_format.Merge(str(my_file.read()), parser_object)
    my_file.close()
    return parser_object

# *******************************************************************************************************************
class AsyncFunction:
    """This is a basic wrapper to create a rotating buffer of loading threads. 
    
    Note that each thread will create a pool of processes to load the actual patch images. 
    """
    def __init__(self, num_process, f):
        self.func = f
        self.pool = ThreadPool(processes=num_process)
    
    def run(self, *args):
        result = self.pool.apply_async(self.func, args)
        return result

# *******************************************************************************************************************
def train_batch_triple_init(image_file, tp):
    """Given training parameters (see above) we initialize two loading threads and return their handles. 
    
    This function runs only once at the beginning of training. 
    """
    
    print ("Init loader")
    # create two loader objects
    lti_1 = LoadTrainImages.Load(tp.train_batch_size, tp.patch_size, tp.solver, tp.image_mean, image_file)
    lti_2 = LoadTrainImages.Load(tp.train_batch_size, tp.patch_size, tp.solver, tp.image_mean, image_file)        

    # We have extra function calls as a legacy from older code. 
    lti_1.useTriple(tp.solver)
    lti_2.useTriple(tp.solver)

    lti_1.useRandomAperture(tp.ra_min_size, tp.ra_max_size, tp.ra_patch_size) # Set up random aperture
    lti_2.useRandomAperture(tp.ra_min_size, tp.ra_max_size, tp.ra_patch_size) # Set up random aperture
    
    lti_1.useLoadPool(tp.pool_procs)
    lti_2.useLoadPool(tp.pool_procs)
    
    lti_1._path_prefix = tp.path_prefix   
    lti_2._path_prefix = tp.path_prefix
    
    # attach loaders objects to its own thread, and run one.     
    f = []
    
    f.append(AsyncFunction(1,lti_1.loadTripleBatch))
    f.append(AsyncFunction(1,lti_2.loadTripleBatch))
    
    r1 = f[0].run()
    
    # return the running loader's handle so we can later fetch results. 
    # Also return the function handles to the threads. 
    return f, r1

# *******************************************************************************************************************
def train_batch_triple(batch_toggle, f, tp, r1):    
    """This will load the images and set them into Caffe's blobs. It calls Caffe for one solver step and returns results. 
    
    We are using a rotating buffer where one thread is told to load images while we grab prior loaded images 
    from the other. 
    """
    
    # Run one of the loader threads to fetch a batch of image patches and labels
    r2                                                  = f[batch_toggle].run()
    
    # From the other thread, get the loaded patches and labels
    image_set_1, image_set_2, image_set_3, label_set, do_exit    = r1.get() 
        
    # Toggle which thread will run next iteration. 
    if batch_toggle == 0:
        batch_toggle = 1
    else:
        batch_toggle = 0
    
    # Get images and labels into Caffe's blobs
    tp.solver.net.blobs['data_1'].data[:,:,:,:]     = image_set_1
    tp.solver.net.blobs['data_2'].data[:,:,:,:]     = image_set_2
    tp.solver.net.blobs['data_3'].data[:,:,:,:]     = image_set_3
    tp.solver.net.blobs['label'].data[:,0,0,0]      = label_set 

    # Run the Caffe solver for one iteration. 
    tp.solver.step(1)
    
    # Get loss and softmax output
    layer_loss              = tp.solver.net.blobs[tp.loss_layer].data
    softmax_output          = tp.solver.net.blobs[tp.softmax_layer].data  
    
    # return outputs, labels, batch loading thread toggle and a handle to the running loader thread
    return layer_loss, softmax_output, label_set, batch_toggle, r2, do_exit

# *******************************************************************************************************************
def test_batch_context_triple(image_file, test_transformer_1, test_transformer_2, test_transformer_3, tp):
    """Run test on the trained network given the testing data set. 
    
    We only test on patches without rotation, flipping and aperture. This way we can measure how much adding these
    widgets detracts from performance on a standard right side up image. 
    
    We do not run all test patches because there are way too many. So, we skip a large number of them. We use an odd
    skip size so we get an even sampling of all classes.  
    """
    
    # Set up all kinds of guys
    net_label       = []
    
    do_exit = False

    assert(image_file.__class__.__name__ == "CompactList")
    
    samples = image_file.length()        
    for _ in range(tp.test_batch_size):
        net_label.append(0)

    print ('********************************************')
    print ("TESTING {} Samples From {}".format(int(samples/tp.test_skip),samples))
    print ('********************************************')
        
    count_list          = []
    correct_list        = []
       
    for _ in range(tp.bin_num):
        count_list.append(0)
        correct_list.append(0)
     
    cv_img              = []

    counter             = 0
    correct_counter     = 0
    incorrect_counter   = 0   
    bx                  = 0
    
    # for all testing samples
    for ii in range(samples):
        
        # We will probably skip most of them with a skip offset like 199
        if (tp.test_skip and ii % tp.test_skip == 0) or not tp.test_skip:

            files, label    = image_file.getFileNames(ii)
            i1              = tp.path_prefix + files[0] 
            i2              = tp.path_prefix + files[1]    
            i3              = tp.path_prefix + files[2]        
 
            # When we run out of samples, finish
            if counter == samples:
                break

            # for each sample, load it in, crop it and apply Caffe's preprocess transformer.
            cv_img                              = cv2.imread(i1)
            if type(cv_img) is not np.ndarray:
                print (">>>>> Bad Test Image {}".format(i1)) 
                do_exit = True
                break
            crop_img                            = cv_img[tp.patch_marg_1:tp.patch_marg_2,tp.patch_marg_1:tp.patch_marg_2,:]
            in_img_1                            = test_transformer_1.preprocess('data_1', crop_img)  
            
            cv_img                              = cv2.imread(i2)
            if type(cv_img) is not np.ndarray:
                print (">>>>> Bad Test Image {}".format(i2)) 
                do_exit = True
                break
            crop_img                            = cv_img[tp.patch_marg_1:tp.patch_marg_2,tp.patch_marg_1:tp.patch_marg_2,:]
            in_img_2                            = test_transformer_2.preprocess('data_2', crop_img) 
            
            cv_img                              = cv2.imread(i3)
            if type(cv_img) is not np.ndarray:
                print (">>>>> Bad Test Image {}".format(i3)) 
                do_exit = True
                break
            crop_img                            = cv_img[tp.patch_marg_1:tp.patch_marg_2,tp.patch_marg_1:tp.patch_marg_2,:]
            in_img_3                            = test_transformer_3.preprocess('data_3', crop_img)
                   
            # Get the sample into Caffe's blobs
            tp.solver.test_nets[0].blobs['data_1'].data[bx,:,:,:]       = in_img_1
            tp.solver.test_nets[0].blobs['data_2'].data[bx,:,:,:]       = in_img_2
            tp.solver.test_nets[0].blobs['data_3'].data[bx,:,:,:]       = in_img_3
            tp.solver.test_nets[0].blobs['label'].data[bx]              = label
            net_label[bx]                                               = float(label)
            
            # once we have loaded the the buffer with a full batch, run testing
            if bx == tp.test_batch_size - 1 or ii == samples - 1:

                # pull the lever and run Caffe forward once
                tp.solver.test_nets[0].forward()
            
                # get the results from Caffe
                gt          = net_label
                smax4       = tp.solver.test_nets[0].blobs[tp.softmax_layer].data
                
                # Process the results. This is a little brutal, but I'm too lazy to make
                # this look clean. Deal with it. 
                for x in range(bx+1):
                
                    # little user feedback on progress. 
                    if counter > 0:
                        if counter%100 == 0:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                    
                        if counter%1000 == 0:
                            c = "{}".format(counter)
                            sys.stdout.write(c)
                            sys.stdout.flush()
                            
                        if counter%10000 == 0:
                            print (' ')
                    
                    # check if the returned label matches the ground truth. 
                    max_bin     = -1
                    max_val     = -1
                    bin_count   = 0
                    for sbin in smax4[x]:
                        if sbin > max_val:
                            max_val = sbin
                            max_bin = bin_count     # my class as a bin construct, not same as bin_size
                        bin_count += 1
                    
                    gt_x    = int(gt[x])            # int ground truth
                    
                    count_list[gt_x]            += 1
                    
                    # is the ground truth the same as the network's returned value?
                    if gt[x] == max_bin:
                        correct_counter             += 1
                        correct_list[gt_x]          += 1
                    else:   
                        incorrect_counter           += 1                           
                    
                    counter = counter + 1    
                    
                    # When we run out of samples, finish
                    if counter == samples:
                        break
                    
                bx                  = 0
                cv_img              = []
            else:
                bx += 1
    
    # Compute percent accuracy. 
    total_samples   = correct_counter + incorrect_counter
    correct_p       = float(correct_counter)/float(total_samples)

    print ('********************************************')
    print (' ')
    print ("TEST : count {} percent correct {}".format(total_samples,correct_p*100.0))
    print (' ')
    print ('********************************************')
    
    return correct_p, do_exit

# *******************************************************************************************************************
def save_plot_data(file_name, x, y):
    """Plot data is preserved between runs so we don't mess up our graphs if we exit or crash
    """
    print("save plot data as: {}.npz".format(file_name))
    np.savez_compressed(file_name, x = x, y = y)  
    
# *******************************************************************************************************************
def load_plot_data(file_name):
    """Plot data is preserved between runs so we don't mess up our graphs if we exit or crash
    """
    print("load plot data as: {}".format(file_name))
    _ = check_file(file_name)
    npz_file    = np.load(file_name) 
       
    x     = npz_file['x']
    y     = npz_file['y']
 
    return x, y

# *******************************************************************************************************************
def mr_plot(layer_loss, i, fig_prop, fig=False, ax=False, fig_file='', title='', tp=None):
    """Plot the accuracy on our testing data. 
    
    This function does a few notable things. We have to hang onto the figure handle because the grapher package
    does not do a good job of collecting stale handles. They just seem to hang around slowing things down. We 
    also save and restore the graph so we can keep a nice clean graph even if we exit training or the program crashes. 
    
    """
    assert(type(tp) != types.NoneType)      # needs to be set
    assert(tp.test_iters > 0)               # This needs to be set in the code, copied over
    assert(i >= 0)
    
    # Should we init or load graph data?
    if fig_file is not '' and os.path.isfile(fig_file + '.npz') and i > tp.test_iters:
        old_x, old_y = load_plot_data(fig_file + '.npz')
    else:
        print('Init new plot data')
        old_x = np.zeros(0)
        old_y = np.zeros(0)
    
    # Delete any i "newer" than the current one for continuity
    # This way if we start over from an earlier state, we have a clean graph
    # by deleting graph data that would essentially be from the future.  
    new_x = np.empty((0))
    new_y = np.empty((0))
                     
    for t in range(old_x.shape[0]):
        if old_x[t] < i:
            new_x       = np.append(new_x, old_x[t])
            new_y       = np.append(new_y, old_y[t])             
            
    new_x       = np.append(new_x, i)
    new_y       = np.append(new_y, layer_loss) 
    
    if fig_file is not '':       
        save_plot_data(fig_file, new_x, new_y)
    
    # pyplot crashes without recovery such as exception handling
    # if it cannot make an X connection. 
    if check_X_is_running():
    
        # only create a figure once.       
        if not fig:
            fig = pylab.figure()
            ax  = fig.add_subplot(111)
        
        pylab.ion()
        
        ax.set_title(title)
        ax.plot(new_x, new_y, fig_prop)
        pylab.draw()
        pylab.show()
        
        # Save our graph and its data
        if fig_file is not '':
            fig.savefig(fig_file)
            
    # return just the figure handles. Note that the data is stored on drive
    return fig, ax

# *******************************************************************************************************************
def vis_square(in_data, title='', fig=False, ax=False, transpose_data=True, fig_file=''):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    
    This is primarily used to visualize the first layer features. 
    """
    
    if transpose_data:
        data = in_data.transpose(0, 2, 3, 1)
    else:
        data = in_data
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    #Only create the figure object once
    if not fig:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
    
    plt.ion()
    
    ax.cla()
    ax.set_title(title)
    ax.imshow(data); ax.axis('off')
    plt.draw()
    plt.show()
    
    # Save the figure to drive
    if fig_file is not '':
        fig.savefig(fig_file)
    
    # return just the figure handles
    return fig, ax

# *******************************************************************************************************************
def create_solver_file(solver_file_name, network_file_name, snapshot_path, condition, use_batch_norm_sched, 
                       use_nccl, snaphot_interval):

    print("**********************************************************************************************************************")
    print("NOTE: Creating Solver File: " + solver_file_name)
    print("**********************************************************************************************************************")

    sf = open(solver_file_name,'w')
    
    net_line        = "net: \"" + network_file_name + "\"\n"
    sf.write(net_line)
    
    snapshot_line   = "snapshot_prefix: \"" + snapshot_path + "/train_val_" + condition + "\"\n"
    sf.write(snapshot_line)

    if use_batch_norm_sched:    
        default = "\
test_iter: 20000\n\
test_interval: 1000000000\n\
test_initialization: false\n\
display: 20\n\
average_loss: 40\n\
lr_policy: \"step\"\n\
gamma: 0.1\n\
stepsize: 300000\n\
max_iter: 750000\n\
base_lr: 0.01\n\
momentum: 0.9\n\
weight_decay: 0.0002\n\
solver_mode: GPU\n\
random_seed: 34234562302122\n" 
    else:
        default = "\
test_iter: 20000\n\
test_interval: 1000000000\n\
test_initialization: false\n\
display: 20\n\
average_loss: 40\n\
lr_policy: \"step\"\n\
gamma: 0.96806001063\n\
stepsize: 10000\n\
max_iter: 1500000\n\
base_lr: 0.00666\n\
momentum: 0.9\n\
weight_decay: 0.0002\n\
solver_mode: GPU\n\
random_seed: 34234562302122\n"  

    sf.write(default)
    
    sf.write("snapshot: {}\n".format(snaphot_interval))
    
    if use_nccl:
        sf.write("layer_wise_reduce: false\n")

    sf.close()

# *******************************************************************************************************************
def create_msub(proj_snapshot_base, log_dir, profile, condition):
    
    log_file                = log_dir               + "/RunTripleContext." + condition + ".log"
    msub_file               = proj_snapshot_base    + "/slurm.msub"
    
    File  = open(msub_file,'w')
    
    preamble = "\
#!/bin/bash\n\
#MSUB -l nodes=1             # use 1 node\n\
#MSUB -l walltime=12:00:00   # ask for 12 hours\n\
#MSUB -q gpgpu               # use the gpgpu partition\n\
#MSUB -A hpcdl               # use my account\n"
    File.write(preamble)

    job_line        = "#MSUB -N {}  # user defined job name\n".format(condition)
    File.write(job_line)
    log_line        = "#MSUB -o {}         # user defined job log file\n".format(log_file)
    File.write(log_line)
               
    more_lines      = "\
# print message that a new run is starting\n\
echo \"Starting new run: $SLURM_JOBID\"\n\
date\n\
\n\
# to create a chain of dependent jobs (optional)\n\
echo \"Submitting dependent job\"\n"
    File.write(more_lines)

    depend_line     = "msub -l depend=$SLURM_JOBID {}\n".format(msub_file)
    File.write(depend_line)
    profile_line    = "source {}\n".format(profile)
    File.write(profile_line)
    run_line        = "python {}/python/RunTripleContextSlurm.py\n\n".format(proj_snapshot_base)
    File.write(run_line)
    
    print("**********************************************************************************************************************")
    print("NOTE: Run me stand alone as: python {}/python/RunTripleContextSlurm.py".format(proj_snapshot_base))
    print("OR:   Run me as a batch as: msub --slurm {}".format(msub_file))
    print("**********************************************************************************************************************")

    
    File.close()
               
# *******************************************************************************************************************
def save_project_state(curr_py_file, state_py_dir):
           
    assert(curr_py_file != state_py_dir)  
    
    lparts = string.split(curr_py_file,'/')   
    
    root_dir = "/"
    
    for i in range(len(lparts) - 1):
        root_dir += (lparts[i] + '/')
        
    if not os.path.isdir(state_py_dir):
        os.mkdir(state_py_dir)
        
    cp_command = "cp -R " + root_dir + " " + state_py_dir
    #print("Backing up project state to {} ".format(state_py_dir))
    #print("Command: {}".format(cp_command))
    os.system(cp_command)
    
# *******************************************************************************************************************
def instantiate_slurm(proj_snapshot_dir, network_file_name,
                      condition, log_dir, profile, snaphot_interval,
                      use_batch_norm_sched, 
                      rank, mp_cond, proc_comm, init_new = False):
    
    snapshot_file           = ""
    
    proj_snapshot_base      = proj_snapshot_dir     + '/slurm.' + condition + '/'
    caffe_snapshot_path     = proj_snapshot_base    + '/caffe_model/'
    solver_file             = proj_snapshot_base    + "solver.prototxt"
    caffe_snapshot_prefix   = caffe_snapshot_path   + "/train_val_" + condition 
    new_network_file        = proj_snapshot_base    + "train_val.prototxt"
    
    # IF YOU START IT WRONG, DELETE THIS FILE
    if os.path.isfile(solver_file) and init_new == False:
        
        my_file = "{}_iter*.solverstate".format(caffe_snapshot_prefix)

        call    = "ls -t1 " + my_file
        load_me = True
        
        try:
            subprocess.check_call(call, shell=True)
        except Exception as e:
            print('Cannot find past solverstate in line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            load_me = False

        if rank == 0:
            print("**********************************************************************************************************************")
            print("NOTE: Using project folder: {}".format(proj_snapshot_base))
            print("**********************************************************************************************************************")

        if load_me:
            files           = subprocess.check_output(call,shell=True)
        
            new_str         = string.split(files,'\n')
        
            snapshot_file   = new_str[0]
            
            if rank == 0:
                print("**********************************************************************************************************************")
                print("NOTE: AUTO LOAD: {}".format(snapshot_file))
                print("**********************************************************************************************************************")
        else:
            if rank == 0:
                print("**********************************************************************************************************************")
                print("NOTE: AUTO LOAD: NEW ... CANNOT FIND {}".format(my_file))
                print("**********************************************************************************************************************")
            
        do_exit = False
    # Create a new project and exit        
    else:        
                
        with mp_cond:
            
            if check_file(network_file_name,rank) == 0: 
                do_exit = True
            else:    
                if rank == 0:
                    print("**********************************************************************************************************************")
                    print("NOTE: Creating project folder: {}".format(proj_snapshot_base))
                    print("**********************************************************************************************************************")
                    print("**********************************************************************************************************************")
                    print("NOTE: Copying network to project as: " + new_network_file)
                    print("**********************************************************************************************************************")
                    
                    if os.path.exists(proj_snapshot_dir):
                        pass
                    else:
                        print("**********************************************************************************************************************")
                        print("NOTE: Creating project directory: " + proj_snapshot_dir)
                        print("**********************************************************************************************************************")
                    
                        os.mkdir(proj_snapshot_dir)
                    
                    # copy this project to its own folder so we preserve it in this state
                    save_project_state(__file__, proj_snapshot_base)
                    # make the solver file
                    create_solver_file(solver_file, new_network_file, caffe_snapshot_path, condition, use_batch_norm_sched, 
                                       use_nccl=True, snaphot_interval=snaphot_interval)
                    # copy the network file into the package
                    shutil.copy(network_file_name, new_network_file)
                    # make the slurm batch msub file
                    create_msub(proj_snapshot_base, log_dir, profile, condition)
                    # Create the place to put our snapshots
                    os.mkdir(caffe_snapshot_path)
                    
                    proc_comm[1] = True
                    mp_cond.notify_all()
                else:
                    if proc_comm[1] == False:
                        mp_cond.wait()
           
        do_exit = True
    
    return solver_file, snapshot_file, do_exit

    

        

