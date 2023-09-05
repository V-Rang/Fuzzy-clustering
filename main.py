import numpy as np
import pandas as pd
import random
from utils import *


#reading in data, getting true cluster ids and extracting data points and respective attributes
df = pd.read_csv("soybean-small.data.csv",header=None)
true_cluster_ids = df.iloc[:,-1].values
df.drop(columns=df.columns[-1],inplace=True)
unique_counts = df.nunique()
cols_to_drop = unique_counts[unique_counts == 1].index
df.drop(columns=cols_to_drop,inplace=True)
X = np.array(df)
Attr_matrix = generate_attribute_matrix(X)

#creating true cluster ids from alphabetic true cluster ids 
cluster_id_dict = dict()
unique_true_ids = np.unique(true_cluster_ids)

for i in range(len(unique_true_ids)):
    cluster_id_dict[unique_true_ids[i]] = int(i)

encoded_cluster_ids = np.zeros(len(true_cluster_ids),dtype=int)
for i in range(len(encoded_cluster_ids)):
    encoded_cluster_ids[i] = cluster_id_dict[true_cluster_ids[i]]

# print(cluster_id_dict)
# print(encoded_cluster_ids)
# print(cluster_id_maker(encoded_cluster_ids))

#parameters
k = 4
n = X.shape[0]
d = X.shape[1]
N = 20
Gmax = 15
alpha = 1.2
beta = 0.1
Pm = 0.01

#create 1 function that runs Gmax times and returns 1 best W
#run above function a certain number of times (100 in paper)

# def global_caller(k,N,X,Gmax,alpha,beta,Pm,Attr_matrix):
#     W = initialization_step(N,k,n)
#     for _ in range(Gmax):
#         W = selection_step(N,W,beta)
#         W = crossover_step(N,X,Attr_matrix,alpha,W)
#         W = mutation_step(N,W,Pm)
        
#         # above gen W has N in first dimension,
#         # Thus each run of Gmax gives N Ws to choose from. 
#         # Select best, implement "elitist" strat


#     return W

#shapes:
#W = (k X n) -> (N,k,n)
#Z = (k X d) -> (N,k,d)
#X = (n X d) -> (n,d)

# W = global_caller(k,N,X,Gmax,alpha,beta,Pm,Attr_matrix)

# print(test_val.shape)

Nruns = 100
cur_best = np.zeros(shape=(1,1))

global_best_Fval  = []
global_best_gamma = []

for i in range(Nruns):
    for j in range(Gmax):
        if(j == 0):
            W = initialization_step(N,k,n)
        else:
            cur_best = cur_best.reshape((1,cur_best.shape[0],cur_best.shape[1])) #shape = (1,k,n)
            # print(cur_best.shape)
            W = initialization_step(N-1,k,n) #shape = (N-1,k,n)
            # print(W.shape)
            # print(W.shape)
            W = np.concatenate((W,cur_best),axis=0) #shape = (N,k,n) ??
            # print(W.shape)

        W = selection_step(N,W,beta)
        W,Z = crossover_step(N,X,Attr_matrix,alpha,W)
        W = mutation_step(N,W,Pm)

        #select best out of the N Ws created post above step
        F_vals = np.zeros(N,dtype=float)

        for p in range(N):
            F_vals[p] = F_calc(W[p],X,Z[p],alpha)
        
        #preserving best of current generation for next generation
        cur_best = W[np.argmin(F_vals)] #shape = k X n
        # cur_cluster_centers = Z_calc_given_W(cur_best,X,Attr_matrix,alpha)
        # cur_cluster_centers = Z[np.argmin(F_vals)]
        # print(cur_best.shape)

        #need values for best of last generation
        if(j == Gmax - 1):
            global_best_Fval.append(np.min(F_vals))
            calc_cluster_ids = cluster_assigner(cur_best)
            global_best_gamma.append(gamma_calc(encoded_cluster_ids,calc_cluster_ids,n))

        # print(encoded_cluster_ids)
        # print(calc_cluster_ids)
        print(f"end of {i}: {j}\n")    


# print(cur_best)
# print(X)
# print(cur_cluster_centers)
# print("function value",global_best_Fval)
# print("gamma values:",global_best_gamma)

k = cur_best.shape[0]
n = cur_best.shape[1]
d = X.shape[1]

# def calc_distance(a,b):
#     n = len(a)
#     dist = 0.
#     for i in range(n):
#         if(a[i] == b[i]): dist += 1.
#     return dist

# F = 0.
# for l in range(k):
#     for i in range(n):
#         F += pow(cur_best[l][i],alpha) * calc_distance(X[i],cur_cluster_centers[l])
# print("self val = ",F)


# print(global_best_gamma)
# print(np.mean(global_best_Fval))
# print(np.mean(global_best_gamma))

one_counter = 0
for i in range(len(global_best_gamma)):
    if(global_best_gamma[i] == 1): one_counter += 1

print(one_counter)

import pickle
file = open("best_W.pkl","wb")
pickle.dump(cur_best,file)

final_ids = cluster_assigner(cur_best)
file = open("first_choice_ids.pkl","wb")
pickle.dump(final_ids,file)

file = open("F_vals.pkl","wb")
pickle.dump(global_best_Fval,file)

file = open("gamma_vals.pkl","wb")
pickle.dump(global_best_gamma,file)



# s_l_index = np.zeros(n,dtype=int)
# for i in range(n):
#   s_l_index[i] = second_largest_index(cur_best[:,i])
# file = open("second_choice_ids.pkl","wb")
# pickle.dump(s_l_index,file)

#constructing second choice cluster ids of the best cluster

# with open("best_W.pkl","wb"):
    # pickle.dump(cur_best,file)
# print("end\n")


        
