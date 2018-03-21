import PrefetchTrain
import signal
import caffe
import numpy as np
import time
import multiprocessing
import NumpyFileList
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
    
    # Path to be appended to each image patch name. 
    path_prefix                 = '/fastraid/datasets/ILSVRC2012/patches_84h_110x110_13x13-blur-ab_compact/'
    # Your caffe network prototxt file 
    #network_file_name           = '/home/nathan/caffe_data/bvlc_alexnet/train_val_Context_84h_Trip_80c_v-weight.prototxt'
    #network_file_name           = '/home/nathan/caffe_data/bvlc_alexnet/train_val_Context_84h_Trip_80c_v-weight-front-back.prototxt'
    #network_file_name           = '/home/nathan/caffe_data/bvlc_alexnet/train_val_Context_84h_Trip_gn-like-2_80c_v-weight.prototxt'
    network_file_name           = '/home/nathan/caffe_data/bvlc_alexnet/train_val_Context_84h_Trip_vs-pad-4_80c_v-weight.prototxt'
    
    # Your temporary solver file.
    solver_file_name            = '/home/nathan/caffe_data/bvlc_alexnet/solver_temp.prototxt'
    # Where to save snapshots
    snapshot_path               = '/home/nathan/caffe_model/'
    # Condition is a label used for graphing, display purposes and saving snap shots
    condition                   = 'AlexNet_Context_vs-pad-4_84h_flip-lr-90-180-270-80c_ZIC-MSA_v-weight'
    # Where to save figures
    fig_root                    = '/home/nathan/results/figures/caffe/'

    # Name of a caffemodel to use to initialize our weights from
    weight_file                 = ''
    # Name of a solverstate file to use to resume from prior training
    snapshot_file               = ''
    
    # Alexnet layer names from the network prototxt file   
    start_layer_vis             = 'conv1'           # Visualize This layer
    softmax_layer               = 'softmax_plain'   # For testing, we need this guy
    loss_layer                  = 'loss'            # Your loss layer
    # Are we using a batch normalized network schedule. For plain CaffeNet, set to False
    use_batch_norm_sched        = True
     
    # ImageNet mean gray
    image_mean                  = [104.0, 117.0, 123.0] # ImageNET
    # Given a 110x110 size patch, what are the range of scales we can resize it to before cropping out 96x96?     
    ra_max_size                 = 128 # Goes to a max size corresponding to an image of 448x448
    ra_min_size                 = 96  # Goes to a min size corresponding to an image of 171x171
    # Training batch size. The script will auto resize this when using more than one GPU
    train_batch_size            = 128 
    # Testing batch size.
    test_batch_size             = 20
    # How many classes you will test over.
    bin_num                     = 20
    # The actual size of the patchs (96x96)
    patch_size                  = 96
    # Tells us where to center crop during testing
    patch_marg_1                = 7
    patch_marg_2                = 110
    # How many iters should we wait to display info?
    display_iters               = 20
    # How many iters should we wait to test the network
    test_iters                  = 5000
    # Smoothing parameter over displayed loss
    loss_lambda                 = 20
    # Stride over the testing data set so we only use a subset. 
    test_skip                   = 199
    # How often to snapshot the solver state
    snaphot_interval            = 50000


    # ******************************************************************************************************************* 
    # ******************************************************************************************************************* 
    # Dont edit after here
    # ******************************************************************************************************************* 
    # ******************************************************************************************************************* 
    if uid:
        use_nccl = True
    else:
        use_nccl = False
        
    # training and testing list files
    test_list_file              = path_prefix + 'val/val_list.nfl.npz'
    train_list_file             = path_prefix + 'train/train_list.nfl.npz'   
    
    with MP_COND:
        if rank == 0:
            PrefetchTrain.create_solver_file(solver_file_name, network_file_name, snapshot_path, condition, 
                                             use_batch_norm_sched, use_nccl, snaphot_interval)
            proc_comm[0] = True
            MP_COND.notify_all()
        else:
            if proc_comm[0] == False:
                MP_COND.wait()
    
    fig_model                   = condition
    fig_name_err                = fig_root + fig_model + '.err.png'  
    fig_name_sqr                = fig_root + fig_model + '.sqr.jpg' 
    fig_prop                    = 'b--'
       
    '''
    We will now configure a bunch of things before we run the main loop. NCCL needs some things to be in a
    particular order. Some tasks are reserved for a single process alone. These always run on the first GPU
    in the list.
    '''
    
    batch_toggle        = 0
    
    print('GPU:{} Set Caffe Device'.format(gpus[rank]))
    
    print('GPU:{} Set Device'.format(gpus[rank]))
    caffe.set_device(gpus[rank])                    ### THIS ALWAYS HAS TO COME BEFORE OTHER CAFFE SETTERS!!!
    
    # Set up multi processing
    if use_nccl:
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
    max_iters           = solver_params.max_iter
    
    print("GPU:{} Adjusted Batch Size For Each GPU : {}".format(gpus[rank],train_batch_size))
    
    # This script does not support iters.       
    assert(solver_params.iter_size < 2)
    
    # Open our training and testing lists, but don't do anything with them yet.         
    print("GPU:{} Loading: {}".format(gpus[rank],test_list_file))
    if rank == 0:
        test_list_in        = open(test_list_file)
    print("GPU:{} Loading: {}".format(gpus[rank],train_list_file))
    train_list_in           = open(train_list_file)
    
    # Do we have a weight file? If so, use it.  
    if weight_file != '':
        print('GPU:{} Loading weight file: {} '.format(gpus[rank],weight_file))
        solver.net.copy_from(weight_file)
    
    # Do we have a snapshot file? If so, use it.     
    if snapshot_file != '':
        print('GPU:{} Loading Snapshot file: {}'.format(gpus[rank],snapshot_file))
        solver.restore(snapshot_file)
    
    if use_nccl:    
        # Create NCCL callback
        nccl = caffe.NCCL(solver, uid)
        nccl.bcast()
        solver.add_callback(nccl)
        
        if solver.param.layer_wise_reduce:
            solver.net.after_backward(nccl)
    
    print("GPU:{} Network and Files Loaded".format(gpus[rank]))
    
    # reshape our training blobs        
    solver.net.blobs['data_1'].reshape(train_batch_size,3,patch_size,patch_size)
    solver.net.blobs['data_2'].reshape(train_batch_size,3,patch_size,patch_size)
    solver.net.blobs['data_3'].reshape(train_batch_size,3,patch_size,patch_size)
    solver.net.blobs['label'].reshape(train_batch_size,1,1,1)
    
    print ("GPU:{} Network Train Blobs Set".format(gpus[rank]))
    
    # reshape testing blobs, but only process will do this. 
    if rank == 0:
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
    
    # Process and load our training data set
    print ("GPU:{} Parse nfl context train list".format(gpus[rank]))
    NFL = NumpyFileList.CompactList()
    NFL.load(train_list_in)
    train_image_file   = NFL

    train_list_in.close()
    
    # process and load our testing data set. Only one GPU will do this. 
    if rank == 0:
        print ("GPU:{} Parse nfl context test list".format(gpus[rank]))
        NFL = NumpyFileList.CompactList()
        NFL.load(test_list_in)
        test_image_file   = NFL

        test_list_in.close()
            
    print ("GPU:{} Lists Parsed".format(gpus[rank]))
    
    # Once we launch the threads, we need to exit gently         
    TRAIN_LOOP = True
    
    # Init the two main loader threads and return handles         
    f, r = PrefetchTrain.train_batch_triple_init(train_image_file, tp)
    
    # set some things we need to set. 
    loss_avg    = 0.0
    cstart      = 0.0
           
    print("GPU:{} PREFETCH TRAIN".format(gpus[rank]))
    
    start_iter  = True
    layer_loss  = 0
    
    vis_fig     = False
    vis_ax      = False
    
    plot_fig    = False
    plot_ax     = False
    
    print("GPU:{} START LOOP".format(gpus[rank]))
    
    '''
    This is our main training loop. From here on out we will stay in this loop until exit. Most of the code here is for
    display and control. train_batch_triple is the only thing that needs to be called to train the network. 
    '''
    while True:
        
        i       = int(solver.iter)       
        display = False
        
        # Do we compute display timing data this iteration?
        if (i % display_iters == 0 or start_iter):
            cend        = time.time()
            timer       = cend - cstart
            cstart      = cend
    
            # It's annoying and useless to print stats like this on the first iter
            if not start_iter:
                t               = timer/float(display_iters)
                # Only once process prints this stuff out. 
                if rank == 0:
                    print("GPU:{} ({}) {} ".format(gpus[rank], i, condition)) 
                    print("GPU:{} Average TIME {}".format(gpus[rank], t))
                
                display = True
        
        # run the actual training step on Caffe. Get back a run handle r and performance data
        layer_loss, _, _, batch_toggle, r = PrefetchTrain.train_batch_triple(batch_toggle, f, tp, r)
        
        # compute a running average over loss
        if start_iter:
            loss_avg = layer_loss
        else:
            loss_avg = (layer_loss + loss_avg*loss_lambda) / (1.0 + loss_lambda)

        avg_guys[rank] = loss_avg 

        # Update the figure showing the first layer filters. Only one process does this. 
        if display and rank == 0:
            vis_fig, vis_ax = PrefetchTrain.vis_square(solver.net.params[start_layer_vis][0].data, condition, vis_fig, vis_ax, True, fig_name_sqr)
        
        # when we reach the right iteration, we will test the network and plot the performance       
        if (rank == 0) or i == int(max_iters):
            
            if (i != 0 and i % test_iters == 0) or i == int(max_iters):
                print ("TESTING")
                # Get weights over
                solver.test_nets[0].share_with(solver.net)
        
                # Run the test network
                correct_p = PrefetchTrain.test_batch_context_triple(test_image_file,
                                                                    test_transformer_1, test_transformer_2, test_transformer_3, tp)
                
                # Plot the results of the test. 
                plot_fig,plot_ax = PrefetchTrain.mr_plot(correct_p, i, fig_prop, plot_fig, plot_ax, fig_name_err, condition, tp = tp) 
        
        # one process will collect and display loss over all GPU processes. 
        if display:
            #print("GPU:{} Average LOSS {}".format(gpus[rank],loss_avg)) 
            if rank == 0:
                avg = 0.0
                for ar in avg_guys:
                    avg += ar
                    
                avg /= len(avg_guys)
                
                print("GPU:{} ALL Average LOSS {}".format(gpus[rank],ar)) 
                
        # Exit when maximum iteration is reached.            
        if i == int(max_iters):
            print ("GPU:{} Reaches Maxed Iters".format(gpus[rank]))
            break
        
        # Exit on ctrl-c 
        if FINISH_EXIT:
            print ("GPU:{} Got CTRL-C. Exiting ...".format(gpus[rank]))
            break        
        
        start_iter = False
    
    # When we exit, we always save the current state. Only one process does this.     
    if rank == 0:
        # just in case
        solver.snapshot()
        
        print ('done : Saving and exiting ...')
    

# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# *********** MAIN THREAD
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 

# We start by attaching a signal handler for ctrl-c (SIGINT) so we can exit and save state whenever. 
signal.signal(signal.SIGINT, ctrl_c_handler)

# We set up N processes for each GPU.
# Change this for how many GPUs you have and want to use. 
# If using just one GPU, you should probably set uid to "False"
#gpus    = [0,1,2,3]
gpus    = [3]
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
proc_comm   = multiprocessing.Array('i',1,lock=True)

proc_comm[0] = False

# Set up each process to run caffe_loop
for rank in range(len(gpus)):
    
    proc = Process(target=caffe_loop, args=(gpus, uid, rank, avg_guys, proc_comm))
    
    proc.start()
    procs.append(proc)

# Start the processes. 
for proc in procs:
    proc.join()   

