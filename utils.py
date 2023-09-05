import numpy as np
import random
import math

#collecting functions


##########################################################
#initialization
def col_summer(A):
    m = A.shape[0]
    n = A.shape[1]
    col_total = np.zeros(n,dtype=float)
    for i in range(n): #traversing cols
        for j in range(m): #traversing rows
            col_total[i] += A[j][i]
        if(col_total[i] == 0): #select random index from 0 to k, set to 1??
            random_index = random.randint(0,m-1)
            A[j][random_index] = 1
    return col_total


def row_summer(A):
    m = A.shape[0]
    n = A.shape[1]
    row_total = np.zeros(m,dtype=float)
    for i in range(m): #traversing rows
        for j in range(n): #traversing cols
            row_total[i] += A[i][j]
    return row_total


def make_legal(A):
    m = A.shape[0]
    n = A.shape[1]
    # row_sum_tot = row_summer(A)
    col_sum_tot = col_summer(A)
    for i in range(m): #row
        for j in range(n): #col
            # if(col_sum_tot[j] == 0): continue #not allowed to have all 0s in a col, sum must be 1, fixed, will not happen
            A[i][j] /= col_sum_tot[j]
    #row sum can only be n if 1 in all the cols, cannot decrease a value to be less
    #than 1 else col sum will become less than 1
    #idea: 
    # there can only be one row that has sum total = n, this would be an edge case
    # when values in all the columns = 1, this would mean that the values in all
    #the other rows corresponding to the column with 1 value are 0. i.e. the matrix
    #W looks like:
    #[0 0 ... 0]
    #[| | ... |]
    #[0 0 ... 0]
    #[1 1 ... 1]
    #[0 0 ... 0]
    #[| | ... |]
    #[0 0 ... 0] 
    #Remember, col sum for each col is supposed to be 1, hence the values in all
    #the other rows = 0.
    #To resolve this, we select a random column, then select a random row in that
    #column, other than the row that has all 1s.
    # We then set the value at [row][ind1] = 0.5 and [row][ind2] = 0.5. This
    #ensures legality of the generated matrix.

    row_sum_tot = row_summer(A)
    for i in range(m):
        if(row_sum_tot[i] == n):
            ind1 = random.randint(0,n-1)

            ind2 = None
            while ind2 is None or ind2 == i:
                ind2 = random.randint(0,m-1)
            
            A[i][ind1] = 0.5
            A[ind2][ind1] = 0.5


# def initialization_step(N,k,n):
#     W = np.zeros(shape=(N,k,n),dtype=float)
#     v = np.zeros(k,dtype=float) 
#     i1 = 0
#     while(i1<N):
#         for i in range(n):
#             for j in range(k):
#                 v[j] = random.random()
#             sum_tot = 0
#             for j in range(k): sum_tot += v[j]
#             for j in range(k):
#                 row = (j*n + i)%k
#                 col = (j*n + i)//k
#                 W[i1][row][col] = v[j]/sum_tot
#         make_legal(W[i1])
#         i1 += 1
    
#     return W

def initializer(X):
  k = X.shape[0]
  n = X.shape[1]
  for i in range(n):
    v = np.zeros(k,dtype=float)
    for j in range(k):
      v[j] = random.random()
    v /= np.sum(v)
    X[:,i] = v
  
  #if created X is legal
  cond = True
  for i in range(k):
    if(np.sum(X[i]) > n):
      cond = False
      break
  if(cond == False): initializer(X)
  if(cond == True): return X

def initialization_step(N,k,n):
    W = np.zeros(shape=(N,k,n),dtype=float)
    for i in range(N):
        W[i] = initializer(W[i])
    return W
    

##########################################################
#selection process
def selection_step(N,W,beta):
    F_vals = np.zeros(N,dtype=float)
    for i in range(N):
        rank = np.linalg.matrix_rank(W[i])
        F_vals[i] = beta * pow((1 - beta),(rank-1))

    total_sum = np.sum(F_vals)
    
    P_vals = np.zeros(N+1,dtype=float)

    for i in range(1,N):
        for j in range(i):
            P_vals[i] += F_vals[j]
        P_vals[i] /= total_sum

    P_vals[-1] = 1

    selection_arr = np.zeros(N,dtype=int)

    for i in range(N):
        v = random.random()
        # for j in range(1,N+1):
            # if(v > P_vals[j-1]  and v < P_vals[j]):
                # break
        # selection_arr[i] = j-1
        for j in range(N):
            if(v > P_vals[j] and v < P_vals[j+1]):
                break
        selection_arr[i] = j
    # print(selection_arr)
        
    W_new = np.zeros(shape=W.shape,dtype=float)
    for i in range(len(selection_arr)):
      W_new[i] = W[selection_arr[i]]

    return W_new

##########################################################
#prepare attribute matrix
def generate_attribute_matrix(X):
    # X is n X d, n -> number of objects
    # d-> dimension of each object, (no of attrbutes of each object)
    # jth attribute -> has nj possible values, so need a list of lists,
    # where each list is of different size
    
    n = X.shape[0]
    d = X.shape[1]

    attr_matrix = []
    for i in range(d):
        attr_matrix.append(list(np.unique(X[:,i])))
    return attr_matrix

    # d = X.shape[1]
    # max_val_array = np.zeros(d,dtype=int)
    # for i in range(d):
    #     max_val_array[i] = max(X[:,i])
    # A = []
    # for i in range(d):
    #     temp_array = [i for i in range(0,max_val_array[i]+1)]
    #     A.append(temp_array)
    # return A

##########################################################
#calculating Z given W
def r_finder(W_arr,X_arr,A_arr,alpha):
    n = len(X_arr)
    # nj = max(A_arr)+1
    nj = len(A_arr)
    sum_array = np.zeros(nj,dtype=float)
    for t in range(nj):
        for i in range(n):
            if(X_arr[i] == A_arr[t]):
                sum_array[t] += pow(W_arr[i],alpha)

    return np.argmax(sum_array)

def Z_calc_given_W(W,X,A,alpha):
    k = W.shape[0]
    d = X.shape[1]
    Z = np.zeros(shape=(k,d),dtype=int)
    for l in range(k):
        for j in range(d):
            r = r_finder(W[l,:],X[:,j],A[j],alpha)
            Z[l][j] = A[j][r]
    return Z


##########################################################
#calculating W given Z
def distance_metric(x,y):
    n = len(x)
    distance = 0
    for i in range(n):
        if(x[i] == y[i]):
            distance += 1
    return distance

# def find_vec_in_matrix(A,x):
#     m = A.shape[0]
#     n = A.shape[1]
#     if(n != len(x)): return False

#     for i in range(m):
#         flag = True
#         if(A[i][0] != x[0]): continue
#         else:
#             for j in range(n):
#                 if(A[i][j] != x[j]): 
#                     flag = False
#                     break
#             if(flag == False): continue
#             return True
#     return False


# def equal_vecs(x,y):
#   n = len(x)
#   if( n != len(y)): return False
#   for i in range(n):
#     if(x[i] != y[i]): return False
#   return True

def equal_vecs(x,y):
    n = len(x)
    if(n != len(y)): return False
    for i in range(n):
        if(x[i] != y[i]): return False
    return True

def find_vec_in_matrix(A,x):
  m = A.shape[0]
  n = A.shape[1]
  if(n != len(x)): return False
  for i in range(m):
    if(equal_vecs(A[i],x)):
      return True
  return False


def W_calc_given_Z(Z,alpha,X):
    k = Z.shape[0]
    n = X.shape[0]
    W = np.zeros(shape=(k,n),dtype=float)
    for l in range(k):
        for i in range(n):
            if(find_vec_in_matrix(Z,X[i])):
                if(equal_vecs(X[i],Z[l])): W[l][i] = 1
                else: W[l][i] = 0
            else:
                sum_val = 0.
                local_distance = distance_metric(X[i],Z[l])
                for h in range(k):
                    sum_val += pow( (local_distance/distance_metric(X[i],Z[h])) , (1/(alpha-1))  )
                # sum_val = pow(sum_val, (1/(alpha-1)) )
                W[l][i] = 1/sum_val
    return W

##########################################################
#crossover
def crossover_step(N,X,Attr_matrix,alpha,W):
    k = W.shape[1]
    d = X.shape[1]
    Z = np.zeros(shape=(N,k,d),dtype=float)
    for t in range(N):
        Z[t] = Z_calc_given_W(W[t],X,Attr_matrix,alpha)
        W[t] = W_calc_given_Z(Z[t],alpha,X)
    return W,Z

##########################################################
#mutation
# def mutation_step(N,W,Pm):
#     k = W.shape[1]
#     n = W.shape[2]
#     for t in range(N):
#         for i in range(n):
#             r = random.random()
#             if(r < Pm):
#                 v = np.zeros(k,dtype=float)
#                 for j in range(k):
#                     v[j] = random.random()
#                 sum_tot = np.sum(v)
#                 for j in range(k):
#                     row = (j*n + i)%k
#                     col = (j*n+i)//k
#                     W[t][row][col] = v[j]/sum_tot
#         make_legal(W[t])
#     return W

def  create_random_vec(p):
    v = np.zeros(p,dtype=float)
    for i in range(p):
        v[i] = random.random()
    v /= np.sum(v)
    return v

def mutation_step(N,W,Pm):
    k = W.shape[1]
    n = W.shape[2]

    for t in range(N):
        for i in range(n):
            r = random.random()
            if(r <= Pm):
                # v = np.zeros(k,dtype=float)
                # for j in range(k):
                #     v[j] = random.random()
                # tot_sum = np.sum(v)
                # v /= tot_sum
                W[t][:,i] = create_random_vec(k)

    return W

##########################################################
#clusterid maker
def cluster_id_maker(a):
    #a is an array that contains cluster ids for all points, we wish
    #to have a dictionary that is of the form:
    #{0: [p1,p2, p3 ...],
    # 1: [p5,p6...]}
    cluster_ids = dict()
    for ind,val in enumerate(a):
        if(val in cluster_ids):
            cluster_ids[val].append(ind)
        else:
            cluster_ids[val] = [ind]    
    return cluster_ids

##########################################################
#gamma calculation
def gamma_calc(true_clusterids, calc_clusterids,n):
  
    true_clusterids = cluster_id_maker(true_clusterids)
    calc_clusterids = cluster_id_maker(calc_clusterids)

#   print(true_clusterids)
#   print(calc_clusterids)

    # for i in calc_clusterids.keys():
        # print(calc_clusterids[i])

    # k1 = len(true_clusterids)
    # k2 = len(calc_clusterids)
    
    k1 = int(max(true_clusterids.keys())+1)
    k2 = int(max(calc_clusterids.keys())+1)

    # k2 = max(calc_clusterids.keys())

#   if(k1 == 0 or k2 == 0): print("error?")

    ncv = np.zeros(shape=(k1,k2),dtype=int)

    # for i in range(k1):
        # for j in range(k2):
    for i in true_clusterids.keys():
        for j in calc_clusterids.keys(): 
        # while(True):
        #     try:
        #         calc_clusterids[j]
        #         break
        #     except KeyError:
        #         for q in range(len(calc_clusterids)):
        #             print(calc_clusterids[q])
        #         print("Value of j = {j}")
        #         exit(4)
    #   if(len(true_clusterids[i]) == 0 or len(calc_clusterids[j]) == 0 ):
        #   ncv[i][j] = 0
    #   else:
        # while True:
        #     try:
        #         calc_clusterids[j]
        #         break
        #     except KeyError:
        #         print("{j} not found")
        #         break

            ncv[int(i)][int(j)] = len(np.intersect1d(true_clusterids[i],calc_clusterids[j]))
    
  #numerator:
    term1_tot = 0.
    for i in range(k1):
        for j in range(k2):
            term1_tot += math.comb(ncv[i][j],2)
    
    term1_tot *= math.comb(n,2)
    term2_part1 = 0
    term2_part2 = 0

    for i in range(k1):
        term2_part1 += math.comb(len(true_clusterids.get(i,[])),2)
  
    for i in range(k2):
        term2_part2 += math.comb(len(calc_clusterids.get(i,[])),2)
  
    term2 = term2_part1 * term2_part2
    num  = term1_tot - term2

  #denominator
    term1_den = 0.
    for i in range(k1):
        term1_den += math.comb(len(true_clusterids.get(i,[])),2)
    
    for i in range(k2):
        term1_den += math.comb(len(calc_clusterids.get(i,[])),2)
  
    term1_den *= 0.5 * math.comb(n,2)
    term1_den -= term2
    den = term1_den

    return num/den

##########################################################
#calc F based on a W
def F_calc(W,X,Z,alpha):
  #W -> k X n
  #X -> n X d
  #Z -> k X d
  k = W.shape[0]
  n = W.shape[1]
  d = X.shape[1]

  F_val = 0.
  for l in range(k):
    for i in range(n):
      F_val += pow(W[l][i],alpha)*distance_metric(X[i],Z[l])
  return F_val

#get cluster id array on the basis of generated W
def cluster_assigner(W):
    #W -> k X n
    k = W.shape[0]
    n = W.shape[1]
    cluster_ids = np.zeros(n,dtype=int)
    for i in range(n):
        cluster_ids[i] = np.argmax(W[:,i])

    return cluster_ids


##########################################################
#get second largest element in array -> for getting second choice cluster id
#of every point
def second_largest_index(arr):
    if len(arr) < 2:
        return "Array should have at least two elements"

    largest = second_largest = float('-inf')
    largest_index = second_largest_index = -1

    for i in range(len(arr)):
        if arr[i] > largest:
            second_largest = largest
            second_largest_index = largest_index
            largest = arr[i]
            largest_index = i
        elif arr[i] > second_largest and arr[i] != largest:
            second_largest = arr[i]
            second_largest_index = i

    if second_largest_index == -1:
        return "There is no second largest element"
    else:
        return second_largest_index