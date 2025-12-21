import sys 
print("✅ Python 版本:", sys.version) 

import numpy as np 
print("✅ NumPy 版本:", np.__version__) 

import pandas as pd 
print("✅ Pandas 版本:", pd.__version__) 

import matplotlib 
print("✅ Matplotlib 版本:", matplotlib.__version__) 

from sklearn import __version__ as sk_version 
print("✅ Scikit-learn 版本:", sk_version) 

print("\nDay 1 环境检查通过！")

