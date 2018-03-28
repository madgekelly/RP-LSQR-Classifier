import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import inv

bc_data = pd.read_csv('c:/BC_DATA/BC_data.txt')

bc_data['diagnosis']=bc_data['diagnosis'].map({'M':1,'B':0})

means_data = bc_data.loc[:, ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean']]

diagnosis_data = bc_data.loc[:, ['diagnosis', 'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean']]

X, Y = diagnosis_data.iloc[:, 1:].values, diagnosis_data.iloc[:, 0].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y )

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)

X_test_norm = mms.transform(X_test)

pca = PCA(n_components = 2)

X_train_pca = pca.fit_transform(X_train_norm)

X_test_pca = pca.transform(X_test_norm)

new_col_1 = np.ones((len(X_train_pca),1))

new_col_2 = np.ones((len(X_test_pca), 1))

A = np.append(X_train_pca, new_col_1, 1 )

K = np.append(X_test_pca, new_col_2, 1)

A_t = np.transpose(A)
 
b = Y_train

c = np.dot(A_t, A)

d = inv(c)

e = np.dot(d, A_t)

a = np.dot(e, b)
  
a_ = a[0]
b_ = a[1]
c_ = a[2]

count = 0

for i in range (0, len(X_test_pca)):
    l = Y_test[i]
    j = np.dot(a, K[i])
    if (j>=0.5 and l==0) or (j<0.5 and l==1):
        count = count + 1
print(count)

x_1 = -0.745012
x_2 = 1.69216
y_1 = (-a_/b_)*x_1 + 0.5
y_2 = (-a_/b_)*x_2 + 0.5

plt.plot([x_1, x_2], [y_1, y_2], color = 'k')

plt.show()
