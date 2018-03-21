import numpy as np
import string

class NatesDict(dict):
    """We use a dictionary to find very basic redundancies in file names.
    """
    def __missing__(self, key):
        return -1
    
class CompactList:
    """This is a simple class to store sample file names in a simple compressed form. 
    
    Using a text file with as many as 25 million file names can take up too much memory. So,
    we split the file names by redundant parts. It's fairly quick and good enough. 
    """    
    def __init__(self):
        """Init various containers
        """
        self._file_root_list    = list()
        self._file_root_dict    = NatesDict() 
        self._file_variant_list = list()
        self._file_variant_dict = NatesDict()
        self._item_root_list    = list() 
        self._item_variant_list = list()
        self._item_class_list   = list()
        self._max_class         = 0
        self._variant_divider   = "."
        self._load              = False
     
    def load(self, file_name):
        """Load in a list and connect to the correct objects
        """
                    
        npz_file                    = np.load(file_name) 
           
        self._item_root_list        = npz_file['item_root_list']
        self._item_variant_list     = npz_file['item_variant_list']
        self._item_class_list       = npz_file['item_class_list']
        self._file_root_list        = npz_file['file_root_list'] 
        self._file_variant_list     = npz_file['file_variant_list']
        self._max_class             = npz_file['max_class']
        self._variant_divider       = npz_file['variant_divider']
        self._load                  = True
        
    def save(self, file_name):
        """Save a list. Note that we do not support editing of already existing lists. 
        """
        assert(self._load == False) # NOT SUPPORTED TO WRITE BACK
        
        np.savez_compressed(file_name,  
                            item_root_list      = self._item_root_list, 
                            item_variant_list   = self._item_variant_list,
                            item_class_list     = self._item_class_list,
                            file_root_list      = self._file_root_list, 
                            file_variant_list   = self._file_variant_list,
                            variant_divider     = self._variant_divider,
                            max_class           = self._max_class)  
        
    def length(self):
        """How many samples are in this list?
        """
        return len(self._item_root_list)
        
    def addVariantWithCheck(self,variant):
        
        v_idx = self._file_variant_dict[variant]
        
        if v_idx == -1:
            v_idx = len(self._file_variant_list)
            self._file_variant_list.append(variant)
            self._file_variant_dict[variant] = v_idx
                
        return v_idx
    
    def addRootWithCheck(self,root):
            
        r_idx = self._file_root_dict[root]
        
        if r_idx == -1:
            r_idx = len(self._file_root_list)
            self._file_root_list.append(root)
            self._file_root_dict[root] = r_idx
            
        return r_idx    
    
    def insertFileNames(self, file_names, item_class):
        """Insert a new sample file name in to the list
        """
        root_list       = []
        variant_list    = []
        
        for file_name in file_names:
                  
            lparts      = string.split(file_name,'/')
        
            rp          = ""
            variants    = [] 
        
            for n in range(len(lparts)-1):
                rp = rp + lparts[n] + "/"    
             
            idx     = self.addRootWithCheck(rp)
            roots   = idx
   
            lparts      = string.split(lparts[len(lparts)-1],self._variant_divider)
                
            for part in lparts:
                idx = self.addVariantWithCheck(part)
                variants.append(idx)

            root_list.append(roots)
            variant_list.append(variants)
        
        if int(item_class) > self._max_class:
            self._max_class = int(item_class)
        
        self._item_root_list.append(root_list)
        self._item_variant_list.append(variant_list)
        self._item_class_list.append(item_class)
        
    def getFileNames(self, idx):
        """Reassemble the file names for samples at the given index
        """   
        roots       = self._item_root_list[idx]
        variants    = self._item_variant_list[idx]
        
        vc          = 0
        file_names  = []
        
        for root in roots:
            
            variant = variants[vc]
        
            file_name = self._file_root_list[root]   
            
            for i in range(len(variant)-1):
                file_name = file_name + self._file_variant_list[variant[i]] + "{}".format(self._variant_divider)
                
            file_name = file_name + self._file_variant_list[variant[len(variant)-1]]
            
            vc += 1
            
            file_names.append(file_name)
        
        return file_names, self._item_class_list[idx]
    
    def parseList(self, list_in, variant_divider = "."): 
        """From a raw list, create our numpy file list
        """
        self._variant_divider = variant_divider
        
        for line in list_in.readlines():
            #split by white spaces
            lparse      = string.strip(line, '\n')
            lparts      = string.split(lparse)
            
            self.insertFileNames(lparts[0:len(lparts)-1], lparts[len(lparts)-1])

  
        
        
        
    