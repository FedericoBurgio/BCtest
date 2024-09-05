import pickle
import pandas as pd
filename='DatasetBC.pkl'
with open(filename, 'rb') as file:
    data = pickle.load(file)

if isinstance(data, pd.DataFrame):
    data.to_csv('output.csv', index=False)

with open(filename, 'rb') as file:
    data = pickle.load(file)

with open('output.txt', 'w') as outfile:
    outfile.write(str(data))
