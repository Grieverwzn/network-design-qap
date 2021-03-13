# @author       Jiawei Lu (jiaweil9@asu.edu)
# @time         2021/1/31 22:13
# @desc         [script description]


from scipy.optimize import linear_sum_assignment
import numpy as np
import time

def GLB_(assignment_mat):
    location_ind, building_ind = linear_sum_assignment(assignment_mat)
    value = assignment_mat[location_ind, building_ind].sum()
    return {'building_ind': building_ind, 'location_ind': location_ind, 'value': value}


n = 90
k = 10000

c = np.random.rand(n, n)
time0 = time.time()
for _ in range(k): r = GLB_(c)
time1 = time.time()
print(f'total time: {time1-time0}')




a = np.random.rand(n)
b = np.random.rand(n)
a_rand_idx = np.random.choice(range(n),int(n/2),replace=False)
b_rand_idx = np.random.choice(range(n),int(n/2),replace=False)

time0 = time.time()

# for _ in range(k):
#     cost = sum(np.sort(a) * np.sort(b)[::-1])

for _ in range(k):
    cost1 = sum(np.sort(a[a_rand_idx]) * np.sort(b[a_rand_idx])[::-1])
    cost2 = sum(a[a_rand_idx] * b[b_rand_idx])
    cost = cost1 + cost2

time1 = time.time()


print(f'total time: {time1-time0}')


