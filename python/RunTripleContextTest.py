import PrefetchTrain
import signal
import caffe
import numpy as np
import time
import multiprocessing
import NumpyFileList
import os
from multiprocessing import Process

MP_COND         = multiprocessing.Condition()
TRAIN_LOOP      = False
FINISH_EXIT     = False
    
# *******************************************************************************************************************  
def ctrl_c_handler(signum, frame):
    """Handle ctrl-c save the current state and exit.
    
    If Caffe has not yet started, we will simply exit. Otherwise, we ask the main loop to 
    exit after the completion of the current iteration. Each caffe loop will pass through this function
    on sigint. The loader processes have a dummy function handle (in another file) so they don't hose since sigint is sent 
    to all processes.  
    """
    global MP_COND
    global TRAIN_LOOP
    global FINISH_EXIT
    
    # probably don't need a lock, but lets keep clean
    with MP_COND:
                
        if rank == 0:
            print("Quade, start the reactor. Free Mars ...")
                
        if TRAIN_LOOP == True:
            FINISH_EXIT = True
        else:
            exit()
            
# *******************************************************************************************************************  
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 

# We set up N processes for each GPU.
# Change this for how many GPUs you have and want to use. 
gpus    = [0]

def caffe_loop(gpus, uid, rank, avg_guys, proc_comm):
    """Main loop for each GPU process. 
    
    At the bottom is the main process which creates each GPU process (this guy). We set up all the parameters here and 
    then run the Caffe loop. NCCL links each GPU process implicitly. So, you will not see semaphores or other similars, 
    but NCCL is doing this in the background when Caffe is called. So for example, all processes will sync up when 
    Caffe step is called (in PrefetchTrain). 
    """
    global MP_COND
    global TRAIN_LOOP
    global FINISH_EXIT
    
    # Where is this project located?
    project_home                = '/g/g17/mundhetn/selfsupervised/'
    # Path to training image set  
    path_prefix                 = '/usr/workspace/wsa/mundhetn/image_temp/patches_84h_110x110_13x13-blur-ab_compact/'
    # You can also get the data here, but you should copy it to a faster location:
    #path_prefix                 = '/p/lscratche/brainusr/datasets/ILSVRC2012/patches_84h_110x110_13x13-blur-ab_compact/'
    
    # Condition is a label used for graphing, display purposes and saving snap shots
    # This can be any valid string, but must by file name friendly. 
    #condition                   = 'AlexNet_STD_VWFB_less-dec-0.75_pad_RGA_myl-1-col'
    condition                   = 'ResCeption_STD_leave-out-skip-2-3_PLAY-HARD-TEST'
    
    # Base for where a lot of files are kept or go such as network files
    caffe_data_dir              = project_home      + '/caffe_data/'
    # Where to save figures
    fig_root                    = project_home      + '/figures/'
    # where to save this project
    proj_snapshot_dir           = project_home      + '/py_proj/'
    # where to save moab files
    log_dir                     = project_home      + '/moab_output/'
    # extra profile to run to set enviroment on node
    profile                     = project_home      + '/scripts/profile.sh'
    # Your caffe network prototxt file 
    #network_file_name           = caffe_data_dir    + '/train_val_AlexNet-Custom_triple.prototxt'
    
    #network_file_name           = '/g/g17/mundhetn/caffe_data/bvlc_alexnet/train_val_Context_84h_Trip_80c_v-weight-front-back_less-dec-0.75_pad.prototxt'
    #network_file_name           = '/g/g17/mundhetn/caffe_data/bvlc_alexnet/train_val_Context_84h_Trip_80c_VWFB_less-dec-0.75_pad_myl-1.prototxt'
    network_file_name           = caffe_data_dir + 'train_val_ResCeption_triple.prototxt'
    
    # Name of a caffemodel to use to initialize our weights from
    weight_file                 = '/g/g17/mundhetn/caffe_data/context_models/train_val_ResCeption_STD_leave-out-skip-2-3_iter_750001.caffemodel'
    #weight_file                 = '/g/g17/mundhetn/selfsupervised/py_proj/slurm.ResCeption_STD_leave-out-skip-2-3/caffe_model/train_val_ResCeption_STD_leave-out-skip-2-3_iter_1000.caffemodel'
    #weight_file                 = ''
    
    # Alexnet layer names from the network prototxt file   
    start_layer_vis             = 'conv1'           # Visualize This layer
    softmax_layer               = 'softmax_plain'   # For testing, we need this guy
    loss_layer                  = 'loss'            # Your loss layer
    
    # Are we using a batch normalized network schedule. For plain CaffeNet, set to False
    use_batch_norm_sched        = True
    # Re-init project files?
    init_new                    = False
     
    # ImageNet mean gray
    image_mean                  = [104.0, 117.0, 123.0] # ImageNET
    # Given a 110x110 size patch, what are the range of scales we can resize it to before cropping out 96x96?     
    ra_max_size                 = 128 # Goes to a max size corresponding to an image of 448x448
    ra_min_size                 = 96  # Goes to a min size corresponding to an image of 171x171
    # Training batch size. The script will auto resize this when using more than one GPU
    train_batch_size            = 1 
    # Testing batch size.
    test_batch_size             = 128
    # How many classes you will test over.
    bin_num                     = 20
    # The actual size of the patchs (96x96)
    patch_size                  = 96
    # Tells us where to center crop during testing
    patch_marg_1                = 7
    patch_marg_2                = 110
    # How many iters should we wait to test the network
    test_iters                  = 5000
    # Stride over the testing data set so we only use a subset. 
    test_skip                   = 0
    # How often to snapshot the solver state
    snaphot_interval            = 5000
    
    # training and testing list files
    test_list_file              = path_prefix + 'train/train_list.nfl.npz' 
    
    leave_out_data              = "Skip"
    leave_out_prop_skip         = 3
    leave_out_prop_skip_start   = 2    
    
    save_test_results           = True
    save_test_file              = proj_snapshot_dir     + '/slurm.' + condition + '/test.csv'

    # ******************************************************************************************************************* 
    # ******************************************************************************************************************* 
    # Dont edit after here
    # ******************************************************************************************************************* 
    # ******************************************************************************************************************* 
    
    # check to make sure files and dirs exist
    if PrefetchTrain.check_file(test_list_file,rank) == 0:        return    
    if PrefetchTrain.check_file(profile,rank) == 0:               return 
    
    if PrefetchTrain.check_dir(path_prefix,rank) == 0:            return
    if PrefetchTrain.check_dir(project_home,rank) == 0:           return
    if PrefetchTrain.check_dir(caffe_data_dir,rank) == 0:         return
    
    # Create some directories if needed
    PrefetchTrain.check_create_dir(log_dir,rank)
    PrefetchTrain.check_create_dir(fig_root,rank)

    solver_file_name, snapshot_file, do_exit = PrefetchTrain.instantiate_slurm(proj_snapshot_dir, network_file_name, 
                                                                               condition, log_dir, profile, snaphot_interval,
                                                                               use_batch_norm_sched, 
                                                                               rank, MP_COND, proc_comm, init_new = init_new)
    
    # We just init-ed the whole thing. Now we exit
    if do_exit:
        return
    
    fig_model                   = condition
       
    '''
    We will now configure a bunch of things before we run the main loop. NCCL needs some things to be in a
    particular order. Some tasks are reserved for a single process alone. These always run on the first GPU
    in the list.
    '''
        
    print('GPU:{} Set Caffe Device'.format(gpus[rank]))
    
    print('GPU:{} Set Device'.format(gpus[rank]))
    caffe.set_device(gpus[rank])                    ### THIS ALWAYS HAS TO COME BEFORE OTHER CAFFE SETTERS!!!
    
    # Set up multi processing
    if uid:
        print('GPU:{} Set Solver Count to {}'.format(gpus[rank],len(gpus)))
        caffe.set_solver_count(len(gpus))
        print('GPU:{} Set Solver Rank to {}'.format(gpus[rank],rank))
        caffe.set_solver_rank(rank)
        print('GPU:{} Set Multiprocess'.format(gpus[rank]))
        caffe.set_multiprocess(True)

    # Use GPU like a civilized human being
    print('GPU:{} Set to Use GPU'.format(gpus[rank]))
    caffe.set_mode_gpu()

    # resize the training batch size by number of GPU's we are using
    train_batch_size /= len(gpus)
    
    print('GPU:{} New Train Batch Size {}'.format(gpus[rank],train_batch_size))
    
    print('GPU:{} Load Network and Files'.format(gpus[rank]))
    print("GPU:{} Solver: {}".format(gpus[rank],solver_file_name))
    
    # Create the Caffe solver and read the solver file so we can use some of its parameters
    solver              = caffe.SGDSolver(solver_file_name)
    solver_params       = PrefetchTrain.read_proto_solver_file(solver_file_name)
    
    print("GPU:{} Adjusted Batch Size For Each GPU : {}".format(gpus[rank],train_batch_size))
    
    # This script does not support iters.       
    assert(solver_params.iter_size < 2)
    
    # Open our training and testing lists, but don't do anything with them yet.         
    print("GPU:{} Loading: {}".format(gpus[rank],test_list_file))
    test_list_in        = open(test_list_file)

    # Do we have a weight file? If so, use it.  
    print('GPU:{} Loading weight file: {} '.format(gpus[rank],weight_file))
    solver.net.copy_from(weight_file)
    
        
    print("GPU:{} Network and Files Loaded".format(gpus[rank]))
    
    # reshape our training blobs        
    solver.net.blobs['data_1'].reshape(train_batch_size,3,patch_size,patch_size)
    solver.net.blobs['data_2'].reshape(train_batch_size,3,patch_size,patch_size)
    solver.net.blobs['data_3'].reshape(train_batch_size,3,patch_size,patch_size)
    solver.net.blobs['label'].reshape(train_batch_size,1,1,1)
    
    print ("GPU:{} Network Train Blobs Set".format(gpus[rank]))
    
    # reshape testing blobs, but only process will do this. 

    solver.test_nets[0].blobs['data_1'].reshape(test_batch_size,3,patch_size,patch_size)
    solver.test_nets[0].blobs['data_2'].reshape(test_batch_size,3,patch_size,patch_size)
    solver.test_nets[0].blobs['data_3'].reshape(test_batch_size,3,patch_size,patch_size)
    solver.test_nets[0].blobs['label'].reshape(test_batch_size,1,1,1)
    
    print ("GPU:{} Network Test Blobs Set".format(gpus[rank]))

    test_transformer_1 = caffe.io.Transformer({'data_1': solver.test_nets[0].blobs['data_1'].data.shape})
    test_transformer_1.set_transpose('data_1', (2,0,1))
    test_transformer_1.set_mean('data_1', np.float32(image_mean)) # mean pixel
    test_transformer_2 = caffe.io.Transformer({'data_2': solver.test_nets[0].blobs['data_2'].data.shape})
    test_transformer_2.set_transpose('data_2', (2,0,1))
    test_transformer_2.set_mean('data_2', np.float32(image_mean)) # mean pixel
    test_transformer_3 = caffe.io.Transformer({'data_3': solver.test_nets[0].blobs['data_3'].data.shape})
    test_transformer_3.set_transpose('data_3', (2,0,1))
    test_transformer_3.set_mean('data_3', np.float32(image_mean)) # mean pixel
      
    print ("GPU:{} Network Test Transformer Set".format(gpus[rank]))
    
    # Set up our training parameters object  
    tp = PrefetchTrain.TrainParams(solver, patch_size, patch_marg_1, patch_marg_2, train_batch_size, test_batch_size, 
                                   bin_num, image_mean, loss_layer, softmax_layer)    
        
    # copy a few more items over into our training parameters object
    tp.path_prefix              = path_prefix
    tp.test_skip                = test_skip
    tp.test_iters               = test_iters
    tp.ra_patch_size            = patch_size
    tp.ra_max_size              = ra_max_size
    tp.ra_min_size              = ra_min_size
    
    tp.leave_out_data               = leave_out_data
    tp.leave_out_prop_skip          = leave_out_prop_skip
    tp.leave_out_prop_skip_start    = leave_out_prop_skip_start
    tp.save_test_results            = save_test_results
    tp.save_test_file               = save_test_file
        
    # process and load our testing data set. Only one GPU will do this. 
    print ("GPU:{} Parse nfl context test list".format(gpus[rank]))
    NFL = NumpyFileList.CompactList()
    NFL.load(test_list_in)
    test_image_file   = NFL
    test_list_in.close()
                    
    # Run the test network
    solver.test_nets[0].share_with(solver.net)
    
    correct_p, do_exit = PrefetchTrain.test_batch_context_triple(test_image_file,
                                                                test_transformer_1, test_transformer_2, test_transformer_3, tp)
 
        

# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# *********** MAIN THREAD
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 

# We start by attaching a signal handler for ctrl-c (SIGINT) so we can exit and save state whenever. 
signal.signal(signal.SIGINT, ctrl_c_handler)

procs   = []

# Get an ID to share between NCCL/GPU processes
# If we set this to False, then we will not use NCCL at all
if len(gpus) > 1:
    uid     = caffe.NCCL.new_uid()
else:
    uid     = False

# This passes some performance data between GPU processes
avg_guys    = multiprocessing.Array('f',len(gpus),lock=False)
# So we can pass messages between processes etc.
proc_comm   = multiprocessing.Array('i',3,lock=True)

proc_comm[0] = False
proc_comm[1] = False
proc_comm[2] = False

# Set up each process to run caffe_loop
for rank in range(len(gpus)):
    
    proc = Process(target=caffe_loop, args=(gpus, uid, rank, avg_guys, proc_comm))
    
    proc.start()
    procs.append(proc)

# Start the processes. 
for proc in procs:
    proc.join()   

