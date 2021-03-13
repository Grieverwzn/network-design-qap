import time
import multiprocessing as mp
import numpy as np
import sys
from queue import PriorityQueue
from assignment import *
sys.setrecursionlimit(10000)


class TreeNode:     # This is the node for tree serch
    def __init__(self, nb_unassigned_buildings, assigned_locations, assigned_buildings, location_status, building_status):
        self.lower_value = None
        self.upper_value = None
        self.assignment_mat_1 = None
        self.assignment_mat_2 = None

        self.nb_unassigned_buildings = nb_unassigned_buildings
        self.assigned_locations = assigned_locations
        self.assigned_buildings = assigned_buildings
        self.dest_location_assignment_status = location_status       # True: available, False: assigned
        self.dest_building_assignment_status = building_status


class BAB:
    def __init__(self, instance, args):
        self.instance = instance
        self.args = args

        self.LB_BFS, self.UB_BFS = None, None
        self.best_solution_BFS_1, self.best_solution_BFS_2 = None, None
        self.best_solution_LCS_1, self.best_solution_LCS_2 = None, None

        # for quick access
        self.branch_list = instance.branch_list
        self.target_relative_gap = args['target_relative_gap']
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


    def calculateGLB(self, node):
        GLB_cost_mat = np.zeros([self.nb_of_orig_building, self.nb_of_orig_location])

        for branch in self.branch_list:
            trans_cost_array_temp = branch.trans_cost_array[node.dest_location_assignment_status]
            flow_array_temp = branch.flow_array[node.dest_building_assignment_status]
            value = sum(np.sort(trans_cost_array_temp) * np.sort(flow_array_temp)[::-1])

            assigned_cost = sum(branch.trans_cost_array[node.assigned_locations] * branch.flow_array[node.assigned_buildings])
            cost_ik = value + assigned_cost
            GLB_cost_mat[branch.i_ind,branch.k_ind] = cost_ik

        lower_solution_1 = Hungarian_1(GLB_cost_mat + self.build_cost_orig_mat)
        assignment_mat_1 = np.zeros([self.nb_of_orig_building, self.nb_of_orig_location])
        assignment_mat_1[lower_solution_1['building_ind'], lower_solution_1['location_ind']] = 1

        build_cost_dest_mat_multiplier = self.build_cost_dest_mat.copy()
        build_cost_dest_mat_multiplier[node.assigned_locations,node.assigned_buildings] -= self.M
        lower_solution_2 = Hungarian_2(build_cost_dest_mat_multiplier)
        assignment_mat_2 = np.zeros([self.nb_of_dest_location, self.nb_of_dest_building])
        assignment_mat_2[lower_solution_2['location_ind'], lower_solution_2['building_ind']] = 1

        node.assignment_mat_1 = assignment_mat_1
        node.assignment_mat_2 = assignment_mat_2

        lv = lower_solution_1['value'] + lower_solution_2['value'] + len(node.assigned_buildings) * self.M
        uv = np.sum(self.flow_mat * np.matmul(np.matmul(assignment_mat_1, self.trans_cost_mat), assignment_mat_2)) + \
            np.sum(self.build_cost_orig_mat * assignment_mat_1) + np.sum(self.build_cost_dest_mat * assignment_mat_2)
        node.lower_value = lv
        node.upper_value = uv


    def solveNode(self, live_node):
        tree_nodes = []
        if live_node.assigned_buildings:
            live_building_id = live_node.assigned_buildings[-1] + 1
        else:
            live_building_id = 0

        dest_building_assignment_status = live_node.dest_building_assignment_status.copy()
        dest_building_assignment_status[live_building_id] = False
        assigned_buildings = live_node.assigned_buildings.copy()
        assigned_buildings.append(live_building_id)

        for i in range(self.nb_of_dest_location):
            dest_location_assignment_status = live_node.dest_location_assignment_status.copy()
            if dest_location_assignment_status[i]:
                dest_location_assignment_status[i] = False
                assigned_locations = live_node.assigned_locations.copy()
                assigned_locations.append(i)

                tree_node = TreeNode(live_node.nb_unassigned_buildings-1,
                                     assigned_locations,
                                     assigned_buildings,
                                     dest_location_assignment_status,
                                     dest_building_assignment_status)
                self.calculateGLB(tree_node)
                tree_nodes.append(tree_node)
        return tree_nodes


    def solveNodes(self, nodes):
        child_node_list = []
        lb, ub = np.inf, self.UB_BFS
        best_node = None

        for live_node in nodes:
            if time.time() > self.valid_time_breadth: break

            tree_nodes = self.solveNode(live_node)
            for tree_node in tree_nodes:
                if tree_node.upper_value < ub:
                    ub = tree_node.upper_value
                    best_node = tree_node

                if tree_node.nb_unassigned_buildings > 1:
                    if tree_node.lower_value <= ub:
                        if tree_node.lower_value < lb: lb = tree_node.lower_value
                        child_node_list.append(tree_node)

        return child_node_list, lb, ub, best_node


    def createRoot(self):
        root = TreeNode(self.nb_of_dest_building, [], [], [True] * self.nb_of_dest_location, [True] * self.nb_of_dest_building)
        self.calculateGLB(root)
        return root


    def finishCurrentIter_BFS(self, branch_iter, number_of_nodes, solver_status):
        GAP = (self.UB_BFS - self.LB_BFS) / self.UB_BFS

        print(f'**BNB-BF iter {branch_iter}: Best Lower bound = ', self.LB_BFS)
        print(f'**BNB-BF iter {branch_iter}: Best Upper bound = ', self.UB_BFS)
        print(f'**BNB-BF iter {branch_iter}: GAP = ', GAP)

        if number_of_nodes == 0:
            print('**BNB-BF branch and bound complete')
            solver_status.value = 1
            return True
        if GAP <= self.target_relative_gap:
            print('**BNB-BF target relative gap reached')
            solver_status.value = 1
            return True
        if time.time() >= self.valid_time_breadth:
            print('**BNB-BF time limit reached')
            return True

        return False


    def solve_breadth(self, solver_status, lock):
        if self.args['threads'] == -1:
            cores = mp.cpu_count()
        else:
            cores = self.args['threads']
        p = mp.Pool(processes=cores)

        self.start_time_breadth = time.time()
        self.valid_time_breadth = self.start_time_breadth + self.time_limit

        root = self.createRoot()
        self.LB_BFS, self.UB_BFS = root.lower_value, root.upper_value

        task_list = [[root]] + [[] for _ in range(cores-1)]
        number_of_nodes = 1
        branch_iter = 0

        while True:
            # new iter
            branch_iter += 1
            print(f'**BNB-BF iter {branch_iter}: nodes {number_of_nodes}')

            # solve nodes
            result_list = p.map(self.solveNodes, task_list)

            # update lb and ub
            result_with_new_lb = min(result_list, key=lambda x: x[1])
            new_lb = result_with_new_lb[1]
            if self.LB_BFS < new_lb < np.inf:
                self.LB_BFS = new_lb

            result_with_new_ub = min(result_list, key=lambda x: x[2])
            new_ub = result_with_new_ub[2]
            if new_ub < self.UB_BFS:
                self.UB_BFS = new_ub
                self.best_solution_BFS_1 = result_with_new_ub[3].assignment_mat_1
                self.best_solution_BFS_2 = result_with_new_ub[3].assignment_mat_2

            # check child nodes
            all_node_list = []
            for result in result_list:
                for node in result[0]:
                    if node.lower_value < self.UB_BFS:
                        all_node_list.append(node)
            number_of_nodes = len(all_node_list)

            # end current iter
            t1 = time.time()
            print(f'**BNB-BF iter {branch_iter}: elapsed time {t1 - self.start_time_breadth}')
            stop_flag = self.finishCurrentIter_BFS(branch_iter, number_of_nodes, solver_status)
            if stop_flag: break

            # prepare next iter
            ave_load = int(np.ceil(number_of_nodes / cores))
            task_list = []
            for i in range(cores-1): task_list.append(all_node_list[i*ave_load:(i+1)*ave_load])
            task_list.append(all_node_list[(i+1)*ave_load:])

        lock.acquire()
        print(f'**BNB-BF Best Solution')
        print('assignment matrix 1')
        print(self.best_solution_BFS_1)
        print('assignment matrix 2')
        print(self.best_solution_BFS_2)
        lock.release()


    def solve_lb(self, solver_status, lock):
        self.start_time_lb = time.time()
        self.valid_time_lb = self.start_time_lb + self.time_limit

        root = self.createRoot()
        lb, ub = root.lower_value, root.upper_value

        pq = PriorityQueue()
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
                    self.best_solution_LCS_1 = tree_node.assignment_mat_1
                    self.best_solution_LCS_2 = tree_node.assignment_mat_2

                if tree_node.nb_unassigned_buildings > 1:
                    if tree_node.lower_value <= ub:
                        pq.put((tree_node.lower_value, node_no, tree_node))
                        node_no += 1

        gap = (ub - lb) / ub
        print(f'--BNB-LB Best Lower bound = {lb}, Best Upper bound = {ub}, GAP = {gap}')
        lock.acquire()
        print(f'**BNB-LB Best Solution')
        print('assignment matrix 1')
        print(self.best_solution_LCS_1)
        print('assignment matrix 2')
        print(self.best_solution_LCS_2)
        lock.release()


    def solve(self):
        solver_status = mp.Manager().Value('i',0)
        lock = mp.Manager().Lock()
        p1 = mp.Process(target=self.solve_breadth, args=(solver_status,lock))
        p2 = mp.Process(target=self.solve_lb, args=(solver_status,lock))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # self.solve_breadth(solver_status)
        # self.solve_lb(0,0)
