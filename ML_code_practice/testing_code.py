import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"C:\Users\suraj\Downloads\Data.csv")

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,3].values

