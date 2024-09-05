# TUM - MIRMI - ATARI lab
# Victor DHEDIN, 2024

from typing import Any, List    
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class DataRecorderAbstract(object):
    def __init__(self,
                 record_dir:str="") -> None:
        self.record_dir = record_dir
        
    def record(self,
               q:np.array,
               v:np.array,
               tau:np.array,
               robot_data:Any,
               **kwargs,
               ) -> None:
        pass


        
   
        
        
        
        