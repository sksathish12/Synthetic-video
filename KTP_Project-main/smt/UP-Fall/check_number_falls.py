import pandas as pd

df_features = pd.read_csv('Features_1&0.5_Vision.csv', header=1)
labels = df_features['Tag']
fall = 0
nofall = 0

for x in labels:
    if x >= 1 and x < 6:
        fall = fall + 1
        
    else:
        nofall = nofall + 1
        
print("Number of falls = ", fall)
print("Number of non-falls = ", nofall)