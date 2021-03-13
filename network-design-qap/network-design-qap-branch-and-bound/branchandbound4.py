import time
import pandas as pd 
import os
from hungarian import *
from collections import deque
import multiprocessing as mp
import numpy as np
import sys
sys.setrecursionlimit(1000000)
sys.getrecursionlimit()

class TreeNode:     # This is the node for tree serch
    def __init__(self, builidng_id, location_id, tree_vec, parent):
        # self.tree_node_id = None
        self.building_id = int(builidng_id)
        self.location_id = int(location_id)
        self.parent = parent
        self.lower_value = None
        self.upper_value = None
        self.tree_vec = tree_vec        # tree vector indicates at the tree nodes the locations are assigned to which building
        self.assignment_mat_1 = None
        self.assignment_mat_2 = None


def GLB_(assignment_mat):
    location_ind, building_ind = linear_sum_assignment(assignment_mat)
    value = assignment_mat[location_ind, building_ind].sum()    # + M # As this is the symmetric case, the i,k of each branch must be chosen in its corresponding l,j
    return {'building_ind': building_ind, 'location_ind': location_ind, 'value': value}


branch_list = []
GLB_cost_mat = np.zeros((1,1))
def init(branch_list_, GLB_cost_mat_):
    global branch_list
    global GLB_cost_mat
    branch_list = branch_list_
    GLB_cost_mat = GLB_cost_mat_


def Branch_update(multiplier_mat, tmp_value):
    for branch in branch_list:
        branch.assignment_mat = branch.init_assignment_mat - multiplier_mat
        solution = GLB_(branch.assignment_mat)
        branch.location_ind = solution['location_ind']
        branch.building_ind = solution['building_ind']
        branch.lower_value = solution['value'] + tmp_value
        GLB_cost_mat[branch.i_ind - 1][branch.k_ind - 1] = branch.lower_value


class BAB:
    def __init__(self, instance, glbsolver, args, cwd):
        self.instance = instance
        self.LB, self.UB = glbsolver.LB, glbsolver.UB
        self.args = args

        self.bf_lower_bound_list = [self.LB]
        self.bf_upper_bound_list = [self.UB]
        self.lb_lower_bound_list = [self.LB]
        self.lb_upper_bound_list = [self.UB]
        self.tree_node_list=[0]

        self.current_layer_nodes = []
        self.branch_iter = 0
        self.best_solution_1 = None
        self.best_solution_2 = None

        self.random_i1_list = []
        self.random_i2_list = []
        self.nb_local = 0

        # for quick access
        self.target_relative_gap = args['target_relative_gap']
        self.max_branch_iters = args['max_branch_iters']
        self.M = args['M']
        self.time_limit = args['time_limit']
        self.start_time_breadth = 0.0
        self.valid_time_breadth = 0.0
        self.start_time_lb = 0.0
        self.valid_time_lb = 0.0
        self.nb_of_orig_building = instance.nb_of_orig_building
        self.nb_of_orig_location = instance.nb_of_orig_location
        self.nb_of_dest_building = instance.nb_of_dest_building
        self.nb_of_dest_location = instance.nb_of_dest_location
        self.flow_mat = instance.flow_mat
        self.trans_cost_mat = instance.trans_cost_mat
        self.build_cost_orig_mat = instance.build_cost_orig_mat
        self.build_cost_dest_mat = instance.build_cost_dest_mat
        self.pathfile=cwd
        


    def local_search(self, tree_node):
        assignment_mat_1, assignment_mat_2 = tree_node.assignment_mat_1, tree_node.assignment_mat_2
        UpperBound = np.sum(self.flow_mat * np.matmul(np.matmul(assignment_mat_1, self.trans_cost_mat), assignment_mat_2)) + \
                     np.sum(self.build_cost_orig_mat * assignment_mat_1) + np.sum(self.build_cost_dest_mat * assignment_mat_2)

        tree_node.upper_value = UpperBound
        return

        local_search_list = deque()
        local_search_list.append(assignment_mat_1)
        Flag_Swap = 1
        while (len(local_search_list) != 0 and Flag_Swap <= 10000):
            temp_assign_mat = local_search_list[0]
            assignment_mat_tmp = local_search_list[0]

            for i in range(self.nb_local):
                temp_assign_mat = local_search_list[0]
                if self.random_i1_list[i] != self.random_i2_list[i]:
                    temp_assign_mat[[self.random_i1_list[i], self.random_i2_list[i]], :] = temp_assign_mat[[self.random_i2_list[i], self.random_i1_list[i]],:]

                tmp_UB = np.sum(self.flow_mat * np.matmul(np.matmul(temp_assign_mat, self.trans_cost_mat), assignment_mat_2)) + \
                         np.sum(self.build_cost_orig_mat * temp_assign_mat) + np.sum(self.build_cost_dest_mat * assignment_mat_2)
                if tmp_UB < UpperBound:
                    # print(UpperBound)
                    UpperBound = tmp_UB
                    assignment_mat_tmp = temp_assign_mat
                    local_search_list.append(assignment_mat_tmp)

            local_search_list.popleft()
            Flag_Swap = Flag_Swap + 1

        assignment_mat_1 = assignment_mat_tmp

        local_search_list = deque()
        local_search_list.append(assignment_mat_2)
        while (len(local_search_list) != 0 and Flag_Swap <= 20000):
            temp_assign_mat = local_search_list[0]
            assignment_mat_tmp = local_search_list[0]

            for i in range(self.nb_local):
                temp_assign_mat = local_search_list[0]
                if self.random_i1_list[i] != self.random_i2_list[i]:
                    temp_assign_mat[[self.random_i1_list[i], self.random_i2_list[i]], :] = temp_assign_mat[[self.random_i2_list[i], self.random_i1_list[i]],:]
                tmp_UB = np.sum(self.flow_mat * np.matmul(np.matmul(assignment_mat_1, self.trans_cost_mat),temp_assign_mat.T)) + \
                         np.sum(self.build_cost_orig_mat * assignment_mat_1) + np.sum(self.build_cost_dest_mat * temp_assign_mat)

                if tmp_UB < UpperBound:
                    UpperBound = tmp_UB
                    assignment_mat_tmp = temp_assign_mat
                    local_search_list.append(assignment_mat_tmp)
            local_search_list.popleft()
            Flag_Swap += 1

        assignment_mat_2 = assignment_mat_tmp
        tree_node.upper_value = UpperBound
        tree_node.assignment_mat_1, tree_node.assignment_mat_2 = assignment_mat_1, assignment_mat_2


    def solveNode(self, live_node):
        tree_nodes = []
        live_building_id = int(live_node.building_id + 1)
        for i in range(self.nb_of_dest_location):
            tmp_tree_vec = live_node.tree_vec.copy()  # should copy, not use ip address
            if tmp_tree_vec[i] == -1:  # and tmp_tree_vec.count(-1) > 1        # todo: change tree_vec to dict
                tmp_tree_vec[i] = live_building_id
                tree_node = TreeNode(live_building_id, i, tmp_tree_vec, live_node)
                multiplier_mat = np.zeros([self.nb_of_dest_location, self.nb_of_dest_building])
                tmp_value = 0
                for k in range(self.nb_of_dest_building):
                    if tree_node.tree_vec[k] != -1:
                        l_ind = k
                        j_ind = tree_node.tree_vec[k]
                        multiplier_mat[l_ind, j_ind] = self.M
                        tmp_value += self.M

                Branch_update(multiplier_mat, tmp_value)

                lower_solution_1 = Hungarian_1(GLB_cost_mat + self.build_cost_orig_mat)
                assignment_mat_1 = np.zeros([self.nb_of_orig_building, self.nb_of_orig_location])
                assignment_mat_1[lower_solution_1['building_ind'], lower_solution_1['location_ind']] = 1

                lower_solution_2 = Hungarian_2(self.build_cost_dest_mat - multiplier_mat)
                assignment_mat_2 = np.zeros([self.nb_of_dest_location, self.nb_of_dest_building])
                assignment_mat_2[lower_solution_2['location_ind'], lower_solution_2['building_ind']] = 1

                tree_node.lower_value = lower_solution_1['value'] + lower_solution_2['value'] + tmp_value
                tree_node.assignment_mat_1, tree_node.assignment_mat_2 = assignment_mat_1, assignment_mat_2
                self.local_search(tree_node)
                tree_nodes.append(tree_node)
        return tree_nodes


    def solveNodes(self, nodes):
        child_node_list = []
        lb, ub = np.inf, self.UB
        best_node = None

        for live_node in nodes:
            if time.time() > self.valid_time_breadth: break

            tree_nodes = self.solveNode(live_node)
            for tree_node in tree_nodes:
                if tree_node.upper_value < ub:
                    ub = tree_node.upper_value
                    best_node = tree_node

                # as still two locations are not assigned, the solution is an lower bound solution
                if tree_node.tree_vec.count(-1) > 1:
                    if tree_node.lower_value <= ub:
                        if tree_node.lower_value < lb: lb = tree_node.lower_value
                        child_node_list.append(tree_node)

        return child_node_list, lb, ub, best_node


    def createRoot(self):
        tree_vec = [-1] * self.nb_of_dest_building
        root = TreeNode(-1, -1, tree_vec, -1)  # generate the root tree_node
        root.lower_value = self.LB
        root.upper_value = self.UB
        return root


    def checkStopCondition(self):
        GAP = (self.UB - self.LB) / self.UB

        print(f'**BNB-BF iter {self.branch_iter}: Best Lower bound = ', self.LB)
        print(f'**BNB-BF iter {self.branch_iter}: Best Upper bound = ', self.UB)
        print(f'**BNB-BF iter {self.branch_iter}: GAP = ', GAP)
        self.bf_lower_bound_list.append(self.LB)
        self.bf_upper_bound_list.append(self.UB)

        

        if GAP <= self.target_relative_gap:
            print('**BNB-BF target relative gap reached')
            return True
        if self.branch_iter >= self.max_branch_iters:
            print('**BNB-BF max branch iters reached')
            return True
        if time.time() >= self.valid_time_breadth:
            print('**BNB-BF time limit reached')
            return True


    def createRandomList(self):
        for i in range(self.nb_of_orig_building):
            for j in range(self.nb_of_orig_building):
                if i != j:
                    self.random_i1_list.append(i)
                    self.random_i2_list.append(j)
        self.nb_local = len(self.random_i1_list)


    def solve_breadth(self, solver_status):
        self.createRandomList()

        if self.args['threads'] == -1:
            cores = mp.cpu_count()
        else:
            cores = self.args['threads']
        p = mp.Pool(processes=cores, initializer=init, initargs=(self.instance.branch_list,self.instance.GLB_cost_mat))

        self.start_time_breadth = time.time()
        self.valid_time_breadth = self.start_time_breadth + self.time_limit

        root = self.createRoot()
        task_list = [[root]] + [[] for _ in range(cores-1)]
        number_of_nodes = 1

        while True:
            # new iter
            self.branch_iter += 1
            print(f'**BNB-BF iter {self.branch_iter}: nodes {number_of_nodes}')
            self.tree_node_list.append(number_of_nodes)
            # solve nodes
            result_list = p.map(self.solveNodes, task_list)

            # update lb and ub
            result_with_new_lb = min(result_list, key=lambda x: x[1])
            new_lb = result_with_new_lb[1]
            if self.LB < new_lb < np.inf:
                self.LB = new_lb

            result_with_new_ub = min(result_list, key=lambda x: x[2])
            new_ub = result_with_new_ub[2]
            if new_ub < self.UB:
                self.UB = new_ub
                self.best_solution_1 = result_with_new_ub[3].assignment_mat_1
                self.best_solution_2 = result_with_new_ub[3].assignment_mat_2

            stop_flag = self.checkStopCondition()
            if stop_flag: break

            # update task_list
            all_node_list = []
            for result in result_list:
                for node in result[0]:
                    if node.lower_value < self.UB:
                        all_node_list.append(node)

            number_of_nodes = len(all_node_list)

            if number_of_nodes == 0:
                print('**BNB-BF branch and bound complete')
                solver_status.value = 1
                break

            ave_load = int(np.ceil(number_of_nodes / cores))
            task_list = []
            for i in range(cores-1): task_list.append(all_node_list[i*ave_load:(i+1)*ave_load])
            task_list.append(all_node_list[(i+1)*ave_load:])

            # time
            t1 = time.time()
            print(f'**BNB-BF iter {self.branch_iter}: elapsed time {t1 - self.start_time_breadth}')

        print(f'**BNB-BF best solution1 {self.best_solution_1}, best solution2 {self.best_solution_2}')
        solution_1_df=pd.DataFrame(self.best_solution_1)
        solution_2_df=pd.DataFrame(self.best_solution_2)
        lb_ub_dict={'LB':self.bf_lower_bound_list,'UB':self.bf_upper_bound_list,'tree_node': self.tree_node_list}
        lb_ub_df=pd.DataFrame(lb_ub_dict)
        solution_1_df.to_csv(os.path.join(self.pathfile,'breadth_first_assignment_1.csv'))
        solution_2_df.to_csv(os.path.join(self.pathfile,'breadth_first_assignment_2.csv'))
        lb_ub_df.to_csv(os.path.join(self.pathfile,'breadth_first_lb_ub_iter.csv'))


    def solve_lb(self, solver_status):
        from queue import PriorityQueue
        import copy

        global branch_list
        global GLB_cost_mat
        branch_list = copy.deepcopy(self.instance.branch_list)
        GLB_cost_mat = copy.deepcopy(self.instance.GLB_cost_mat)

        self.start_time_lb = time.time()
        self.valid_time_lb = self.start_time_lb + self.time_limit

        lb, ub = self.LB, self.UB
        best_solution_1, best_solution_2 = None, None

        pq = PriorityQueue()
        root = self.createRoot()
        node_no = 0
        pq.put((root.lower_value, node_no, root))
        node_no += 1

        while (pq.queue):
            if solver_status.value == 1:
                print('--BNB-LB stopped as BNB-BF has completed')
                break

            if time.time() > self.valid_time_lb:
                print('--BNB-LB time limit reached')
                break

            lower_value, _, live_node = pq.get()
            if lower_value > ub:
                print('--BNB-LB branch and bound complete')
                break
            lb = lower_value

            tree_nodes = self.solveNode(live_node)
            for tree_node in tree_nodes:
                if tree_node.upper_value < ub:
                    ub = tree_node.upper_value
                    best_solution_1 = tree_node.assignment_mat_1
                    best_solution_2 = tree_node.assignment_mat_2

                # as still two locations are not assigned, the solution is an lower bound solution
                if tree_node.tree_vec.count(-1) > 1:
                    if tree_node.lower_value <= ub:
                        pq.put((tree_node.lower_value, node_no, tree_node))
                        node_no += 1
            self.lb_lower_bound_list.append(lb)
            self.lb_upper_bound_list.append(ub)
            
        gap = (ub - lb) / ub
        print(f'--BNB-LB: lb = {lb}, ub = {ub}, gap = {gap}')
        print(f'--BNB-LB best solution1 {best_solution_1}, best solution2 {best_solution_2}')


        best_solution_1_df=pd.DataFrame(best_solution_1)
        best_solution_2_df=pd.DataFrame(best_solution_2)
        lb_ub_dict={'LB':self.lb_lower_bound_list,'UB':self.lb_upper_bound_list}
        lb_ub_df=pd.DataFrame(lb_ub_dict)
        best_solution_1_df.to_csv(os.path.join(self.pathfile,'depth_first_assignment_1.csv'))
        best_solution_2_df.to_csv(os.path.join(self.pathfile,'depth_first_assignment_2.csv'))
        lb_ub_df.to_csv(os.path.join(self.pathfile,'depth_first_lb_ub_iter.csv'))


    def solve(self):
        solver_status = mp.Manager().Value('i',0)
        p1 = mp.Process(target=self.solve_breadth, args=(solver_status,))
        p2 = mp.Process(target=self.solve_lb, args=(solver_status,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
