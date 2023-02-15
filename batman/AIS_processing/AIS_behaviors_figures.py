
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
path = None

for file in os.listdir(path):
    if 'AIS_harvest' in file:
        pass
    elif file.endswith( '.feather' ):
        df = pd.read_feather( os.path.join( path, file ))
        break
##
df.columns
##
