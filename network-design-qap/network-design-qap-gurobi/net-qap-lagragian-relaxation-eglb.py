# coding: utf-8

# In[1]:
from gurobipy import *
import pandas as pd
import multiprocessing
from collections import deque

import csv
#from gurobipy import gurobipy

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

# In[2]: Hash table
g_internal_node_seq_no_dict = {}
g_node_id_dict = {}
g_internal_agent_seq_no_dict = {}
g_branch_id_pair_dict = {}

# In[3]: Parameters
g_nb_of_nodes = 0
g_nb_of_links = 0
g_nb_of_agents = 0
g_nb_of_branch = 0
g_nb_of_orig_building = 0
g_nb_of_orig_location = 0
g_nb_of_dest_building = 0
g_nb_of_dest_location = 0

# In[4]: Dictionaries for objects
g_node_list = {}
g_link_list = {}
g_agent_list = {}
g_branch_list = {}

# In[5]: Global arrays
g_flow_mat = None
g_trans_cost_mat = None
g_build_cost_orig_mat = None
g_build_cost_dest_mat = None
g_GLB_cost_mat = None
g_lag_multiplier_mat = None
g_sub_gradient_mat = None
g_assignment_count_mat=None
g_assignment_mat_1 = None
g_assignment_mat_2 = None

# In[6]: Network information
g_building_orig_list = []
g_building_dest_list = []
g_location_orig_list = []
g_location_dest_list = []
g_sub_gradient_list=[]
g_lag_multiplier_list=[]
g_lower_bound_list=[]
g_upper_bound_list=[]

# neos-4 : nitial_rho = 15; rho_step_size = 0.1 ; step_size = 0.01
# Parameter of optimization
node_id_unit = 1000
building_orig = 1000
location_orig = 2000
location_dest = 3000
building_dest = 4000
#M = 1E+10 # Important parameter in symmetric case
UB=float(np.inf)
LB=float(-np.inf)
nb_local = 1000
nb_iter = 200
cwd = r'..\Neos_9_2'
step_size=2
objective_value=0


# Classes
class Node :
    def __init__(self, node_name, node_id) :
        self.name = node_name
        self.node_id = int(node_id)
        self.node_seq_no = 0
        self.m_outgoing_link_list = []
        self.m_incoming_link_list = []
        self.Initialization()

    def Initialization(self) :
        global g_nb_of_nodes
        g_internal_node_seq_no_dict[self.node_id] = g_nb_of_nodes
        g_node_id_dict[g_nb_of_nodes] = self.node_id
        self.node_seq_no = g_nb_of_nodes
        g_nb_of_nodes += 1
        if self.name == 'building node1' :
            g_building_orig_list.append(self.node_id)
        if self.name == 'building node2' :
            g_building_dest_list.append(self.node_id)
        if self.name == 'location node1' :
            g_location_orig_list.append(self.node_id)
        if self.name == 'location node2' :
            g_location_dest_list.append(self.node_id)


class Link :
    def __init__(self, link_id, link_type, from_node_id, to_node_id, built_cost, trans_cost, ) :
        self.link_id = int(link_id)
        self.link_type = int(link_type)
        self.from_node_id = int(from_node_id)
        self.to_node_id = int(to_node_id)
        self.built_cost = float(built_cost)
        self.trans_cost = float(trans_cost)
        self.multiplier = float(0)
        self.Initialization()  # self.CalculateBPRFunctionAndCost()

    def Initialization(self) :
        global g_nb_of_links
        self.from_node_seq_no = g_internal_node_seq_no_dict[self.from_node_id]
        self.to_node_seq_no = g_internal_node_seq_no_dict[self.to_node_id]
        self.link_seq_no = g_nb_of_links
        g_node_list[self.from_node_seq_no].m_outgoing_link_list.append(self)
        g_node_list[self.to_node_seq_no].m_incoming_link_list.append(self)
        g_nb_of_links += 1


class Agent :
    def __init__(self, agent_id, origin_node_id, destination_node_id, customized_cost_link_type,
                 customized_cost_link_value,
                 agent_type, set_of_allowed_link_types) :
        self.agent_id = int(agent_id)
        self.agent_type = agent_type
        self.agent_list_seq_no = 0
        self.origin_node_id = int(origin_node_id)
        self.destination_node_id = int(destination_node_id)
        self.origin_node_seq_no = g_internal_node_seq_no_dict[self.origin_node_id]
        self.destination_node_seq_no = g_internal_node_seq_no_dict[self.destination_node_id]
        self.customized_cost_link_type = customized_cost_link_type
        self.flow = float(customized_cost_link_value)
        self.set_of_allowed_links_types = list(map(int, set_of_allowed_link_types.split(";")))
        self.path_cost = 0
        self.path_node_seq_no_list = []
        self.path_link_seq_no_list = []
        self.path_node_seq_str = ''
        self.path_time_seq_str = ''
        self.number_of_nodes = 0
        self.path_cost = 0
        self.m_path_link_seq_no_list_size = 0
        self.m_current_link_seq_no = 0

        self.Initialization()

    def Initialization(self) :
        global g_nb_of_agents
        if (self.origin_node_id not in g_node_id_dict.values()) or (
                self.destination_node_id not in g_node_id_dict.values()) :
            print('agent', self.agent_id, 'origin or destination node does not exist in node set, please check!')
        else :
            self.agent_list_seq_no = g_nb_of_agents
            g_internal_agent_seq_no_dict[self.agent_id] = g_nb_of_agents
            g_nb_of_agents += 1


class Branch :  # GLB
    def __init__(self, from_node_id, to_node_id) :
        self.branch_id = None
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.lower_value = 0
        self.upper_value = np.inf
        self.init_assignment_mat = None
        self.assignment_mat = None
        self.location_ind = None
        self.building_ind = None
        self.assignment_count = None
        self.Initialization()

    def Initialization(self) :
        global g_nb_of_branch
        i_ind = np.mod(self.from_node_id, node_id_unit)
        k_ind = np.mod(self.to_node_id, node_id_unit)
        self.init_assignment_mat = np.matmul(g_trans_cost_mat[k_ind - 1 :k_ind].T, g_flow_mat[i_ind - 1 :i_ind])
        #self.init_assignment_mat[k_ind - 1, i_ind - 1] = -M
        self.branch_id = g_nb_of_branch
        g_branch_id_pair_dict[(self.from_node_id, self.to_node_id)] = self.branch_id
        g_nb_of_branch += 1


# In[2]:
def g_ReadInputData() :
    print('Start input data...')
    t0 = datetime.datetime.now()
    global g_nb_of_nodes
    global g_nb_of_links
    global g_nb_of_agents
    

    # (1) input the data from csv files
    node_df = pd.read_csv(os.path.join(cwd,'input_node.csv'), encoding='gbk')
    link_df = pd.read_csv(os.path.join(cwd,'input_link.csv'), encoding='gbk')
    agent_df = pd.read_csv(os.path.join(cwd,'input_agent.csv'), encoding='gbk')

    # (2) create  classes
    global g_node_list
    global g_link_list
    global g_agent_list
    global g_branch_list
    g_node_list = node_df.apply(lambda x : Node(node_id=x.node_id, node_name=x.node_name), axis=1)
    print('reading', g_nb_of_nodes, 'nodes...')

    g_link_list = link_df.apply(lambda x : Link(link_id=x.link_id,
                                                from_node_id=x.from_node_id,
                                                to_node_id=x.to_node_id,
                                                built_cost=x.built_cost,
                                                trans_cost=x.trans_cost,
                                                link_type=x.link_type), axis=1)

    print('reading', g_nb_of_links, 'links...')
    g_agent_list = agent_df.apply(lambda x : Agent(agent_id=x.agent_id,
                                                   agent_type=x.agent_type,
                                                   origin_node_id=x.origin_node_id,
                                                   destination_node_id=x.destination_node_id,
                                                   customized_cost_link_type=x.customized_cost_link_type,
                                                   customized_cost_link_value=x.customized_cost_link_value,
                                                   set_of_allowed_link_types=x.set_of_allowed_link_types), axis=1)

    print('reading', g_nb_of_agents, 'agents...')

    t1 = datetime.datetime.now()
    print('Input Data using time', t1 - t0)


# In[2]:
def g_Initialization() :
    print('Start initialization...')
    t1 = datetime.datetime.now()
    global g_nb_of_orig_building
    global g_nb_of_orig_location
    global g_nb_of_dest_building
    global g_nb_of_dest_location
    global g_flow_mat
    global g_trans_cost_mat
    global g_build_cost_orig_mat
    global g_build_cost_dest_mat
    global g_GLB_cost_mat
    global g_lag_multiplier_mat
    global g_sub_gradient_mat

    g_nb_of_orig_building = len(g_building_orig_list)
    g_nb_of_orig_location = len(g_location_orig_list)
    g_nb_of_dest_building = len(g_building_dest_list)
    g_nb_of_dest_location = len(g_location_dest_list)

    g_flow_mat = np.zeros([g_nb_of_orig_building, g_nb_of_dest_building])
    g_trans_cost_mat = np.zeros([g_nb_of_orig_location, g_nb_of_dest_location])
    g_build_cost_orig_mat = np.zeros([g_nb_of_orig_building, g_nb_of_orig_location])
    g_build_cost_dest_mat = np.zeros([g_nb_of_dest_location,g_nb_of_dest_building])
    g_GLB_cost_mat = np.zeros([g_nb_of_orig_building, g_nb_of_orig_location])
    g_lag_multiplier_mat = np.zeros([g_nb_of_dest_location, g_nb_of_dest_building])
    g_sub_gradient_mat = np.zeros([g_nb_of_dest_location, g_nb_of_dest_building])

    for l in range(g_nb_of_links) :
        if (g_link_list[l].link_type == 1) or (g_link_list[l].link_type == 4):  # transportation link
            k_ind = np.mod(g_link_list[l].from_node_id, node_id_unit)
            l_ind = np.mod(g_link_list[l].to_node_id, node_id_unit)
            g_trans_cost_mat[k_ind - 1][l_ind - 1] = g_link_list[l].trans_cost
        if g_link_list[l].link_type == 2 :  # 'building_orig':
            i_ind = np.mod(g_link_list[l].from_node_id, node_id_unit)
            k_ind = np.mod(g_link_list[l].to_node_id, node_id_unit)
            g_build_cost_orig_mat[i_ind - 1][k_ind - 1] = g_link_list[l].built_cost
        if g_link_list[l].link_type == 3 :  # 'building_dest':
            l_ind = np.mod(g_link_list[l].from_node_id, node_id_unit)
            j_ind = np.mod(g_link_list[l].to_node_id, node_id_unit)
            g_build_cost_dest_mat[l_ind - 1][j_ind - 1] = g_link_list[l].built_cost

    for l in range(g_nb_of_agents) :
        #if g_agent_list[l].agent_type == 1 :  # transportation link
        i_ind = np.mod(g_agent_list[l].origin_node_id, node_id_unit)
        j_ind = np.mod(g_agent_list[l].destination_node_id, node_id_unit)
        g_flow_mat[i_ind - 1][j_ind - 1] = g_agent_list[l].flow

    input_list = []
    for l in range(g_nb_of_links) :
        if g_link_list[l].link_type == 2 :  # 'building_orig'
            branch = Branch(g_link_list[l].from_node_id, g_link_list[l].to_node_id)
            branch.assignment_mat = branch.init_assignment_mat  # in the first iteration, intial assignment matrices are used
            input_list.append((branch.branch_id, branch.assignment_mat))
            g_branch_list[branch.branch_id] = branch
            g_branch_list[branch.branch_id].assignment_count = np.zeros([g_nb_of_dest_location,g_nb_of_dest_building])

    for solution in pool.map(GLB, input_list) :
        # print(solution)
        g_branch_list[solution['branch_seq']].lower_value = solution['value']
        g_branch_list[solution['branch_seq']].location_ind = solution['location_ind']
        g_branch_list[solution['branch_seq']].building_ind = solution['building_ind']
        for i in range(g_nb_of_dest_location):
            xx=g_branch_list[solution['branch_seq']].location_ind[i]
            yy=g_branch_list[solution['branch_seq']].building_ind[i]
            g_branch_list[solution['branch_seq']].assignment_count[xx][yy] += 1

    for i in range(g_nb_of_branch) :
        i_ind = np.mod(g_branch_list[i].from_node_id, node_id_unit)
        k_ind = np.mod(g_branch_list[i].to_node_id, node_id_unit)
        g_GLB_cost_mat[i_ind - 1][k_ind - 1] = g_branch_list[i].lower_value

    t2 = datetime.datetime.now()
    print('Initialization using time', t2 - t1)


def Branch_update():
    input_list = []
    for b in range(g_nb_of_branch):
        g_branch_list[b].assignment_mat = g_branch_list[b].init_assignment_mat-g_lag_multiplier_mat
        input_list.append((g_branch_list[b].branch_id, g_branch_list[b].assignment_mat))
        g_branch_list[b].assignment_count = np.zeros([g_nb_of_dest_location, g_nb_of_dest_building])

    for solution in pool.map(GLB, input_list) :
        # print(solution)
        g_branch_list[solution['branch_seq']].location_ind = solution['location_ind']
        g_branch_list[solution['branch_seq']].building_ind = solution['building_ind']
        for i in range(g_nb_of_dest_location) :
            xx = g_branch_list[solution['branch_seq']].location_ind[i]
            yy = g_branch_list[solution['branch_seq']].building_ind[i]
            g_branch_list[solution['branch_seq']].assignment_count[xx][yy] += 1

        g_branch_list[solution['branch_seq']].lower_value = solution['value']

    for b in range(g_nb_of_branch) :
        i_ind = np.mod(g_branch_list[b].from_node_id, node_id_unit)
        k_ind = np.mod(g_branch_list[b].to_node_id, node_id_unit)
        g_GLB_cost_mat[i_ind - 1][k_ind - 1] = g_branch_list[b].lower_value


# def Hungarian(assignment_mat) :
#     #location_ind, building_ind = linear_sum_assignment(assignment_mat)
#     building_ind, location_ind = linear_sum_assignment(assignment_mat)
#     #value = assignment_mat[location_ind, building_ind].sum()
#     value = assignment_mat[building_ind, location_ind].sum()
#     return {'building_ind' : building_ind, 'location_ind' : location_ind, 'value' : value}

def Hungarian_1(assignment_mat) :
    #location_ind, building_ind = linear_sum_assignment(assignment_mat)
    building_ind, location_ind = linear_sum_assignment(assignment_mat)
    #value = assignment_mat[location_ind, building_ind].sum()
    value = assignment_mat[building_ind, location_ind].sum()
    return {'building_ind' : building_ind, 'location_ind' : location_ind, 'value' : value}

def Hungarian_2(assignment_mat) :
    location_ind, building_ind = linear_sum_assignment(assignment_mat)
    #building_ind, location_ind = linear_sum_assignment(assignment_mat)
    value = assignment_mat[location_ind, building_ind].sum()
    #value = assignment_mat[building_ind, location_ind].sum()
    return {'building_ind' : building_ind, 'location_ind' : location_ind, 'value' : value}

def GLB(input_list) :
    assignment_mat = input_list[1]
    branch_seq = input_list[0]
    location_ind, building_ind = linear_sum_assignment(assignment_mat)
    value = assignment_mat[location_ind, building_ind].sum() #+ M # As this is the symmetric case, the i,k of each branch must be chosen in its corresponding l,j
    return {'branch_seq' : branch_seq, 'building_ind' : building_ind, 'location_ind' : location_ind, 'value' : value}


def Trace(lower_solution_1):
    assignment_count_mat = np.zeros([g_nb_of_dest_location,g_nb_of_dest_building])
    for l in range(g_nb_of_orig_location) :
        k_ind = lower_solution_1['location_ind'][l]
        i_ind = lower_solution_1['building_ind'][l]
        branch_id=g_branch_id_pair_dict[(i_ind+1+building_orig, k_ind+1+location_orig)]
        #print((i_ind,k_ind))
        #print(g_branch_list[branch_id].assignment_count)
        assignment_count_mat += g_branch_list[branch_id].assignment_count

    return assignment_count_mat

def local_search():
    global g_assignment_mat_1
    global g_assignment_mat_2
    UpperBound = np.sum(
        g_flow_mat * np.matmul(np.matmul(g_assignment_mat_1, g_trans_cost_mat), g_assignment_mat_2.T)) + np.sum(
        g_build_cost_orig_mat * g_assignment_mat_1) + np.sum(g_build_cost_dest_mat * g_assignment_mat_2)
    local_search_list=deque()
    local_search_list.append(g_assignment_mat_1)
    #temp_assign_mat=g_assignment_mat
    Flag_Swap=1
    while (len(local_search_list)!=0 and Flag_Swap<=10000):
        temp_assign_mat = local_search_list[0]
        g_assignment_mat_tmp = local_search_list[0]
        N = np.size(temp_assign_mat,0)
        random_i1_list=[]
        random_i2_list=[]
        for i in range(N):
            for j in range(N):
                if i!=j:
                    random_i1_list.append(i)
                    random_i2_list.append(j)

        nb_local = len(random_i1_list)
        for i in range(nb_local):
            temp_assign_mat = local_search_list[0]
            if random_i1_list[i]!= random_i2_list[i]:
                temp_assign_mat[[random_i1_list[i],random_i2_list[i]],:] = temp_assign_mat[[random_i2_list[i],random_i1_list[i]],:]

            tmp_UB = np.sum(g_flow_mat * np.matmul(np.matmul(temp_assign_mat, g_trans_cost_mat), g_assignment_mat_2.T)) + np.sum(g_build_cost_orig_mat * temp_assign_mat) + np.sum(g_build_cost_dest_mat * g_assignment_mat_2)
            if tmp_UB < UpperBound:
                #print(UpperBound)
                UpperBound = tmp_UB
                g_assignment_mat_tmp = temp_assign_mat
                local_search_list.append(g_assignment_mat_tmp)

        local_search_list.popleft()
        Flag_Swap = Flag_Swap + 1
        #print(Flag_Swap)

    g_assignment_mat_1 = g_assignment_mat_tmp

    local_search_list=deque()
    local_search_list.append(g_assignment_mat_2)
    while (len(local_search_list)!=0 and Flag_Swap<=20000):
        temp_assign_mat = local_search_list[0]
        g_assignment_mat_tmp = local_search_list[0]
        N = np.size(temp_assign_mat,0)
        random_i1_list=[]
        random_i2_list=[]
        for i in range(N):
            for j in range(N):
                if i!=j:
                    random_i1_list.append(i)
                    random_i2_list.append(j)

        nb_local = len(random_i1_list)
        for i in range(nb_local):
            temp_assign_mat = local_search_list[0]
            if random_i1_list[i]!= random_i2_list[i]:
                temp_assign_mat[[random_i1_list[i],random_i2_list[i]],:] = temp_assign_mat[[random_i2_list[i],random_i1_list[i]],:]
            tmp_UB = np.sum(
        g_flow_mat * np.matmul(np.matmul(g_assignment_mat_1, g_trans_cost_mat), temp_assign_mat.T)) + np.sum(
        g_build_cost_orig_mat * g_assignment_mat_1) + np.sum(g_build_cost_dest_mat * temp_assign_mat)

            if tmp_UB < UpperBound:
                UpperBound = tmp_UB
                g_assignment_mat_tmp = temp_assign_mat
                local_search_list.append(g_assignment_mat_tmp)
        local_search_list.popleft()
        Flag_Swap +=1

    g_assignment_mat_2 = g_assignment_mat_tmp


    return UpperBound, g_assignment_mat_1, g_assignment_mat_2

def Cuttingplane(g_sub_gradient_list,g_lower_bound_list,g_lag_multiplier_list,UB,LB):
    global objective_value
    # Create optimization model
    enviroment = gurobipy.Env()
    enviroment.setParam('TimeLimit', 240)
    model = Model("cutting plane", env=enviroment)
    # Create variables
    multiplier = model.addVars(range(g_nb_of_dest_location), range(g_nb_of_dest_building), name='multiplier', lb=-10, ub=10)
    ZZ = model.addVar(vtype=GRB.CONTINUOUS,name='ZZ')
    nb_constraints = len(g_sub_gradient_list)

    for i in range(nb_constraints):
        tmp_dual = g_lower_bound_list[i]
        tmp_multiplier = g_lag_multiplier_list[i]
        tmp_sug_gradient= g_sub_gradient_list[i]
        model.addConstr(ZZ <= tmp_dual+quicksum((multiplier[l,j]-tmp_multiplier[l,j])*tmp_sug_gradient[l,j]
                                                for l in range(g_nb_of_dest_location) for j in range(g_nb_of_dest_building)))
    model.addConstr(ZZ <= UB)
    model.addConstr(ZZ >= LB)

    model.setObjective(ZZ,GRB.MAXIMIZE)
    model.optimize()
    if model.Status==2:
        solution=model.getAttr('x',multiplier)
        for key in solution.keys():
            g_lag_multiplier_mat[key[0],key[1]] = solution[key]
        objective_value=model.objVal
    return objective_value


# In[10]:
if "__main__" == __name__ :
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    g_ReadInputData()
    g_Initialization()
    
    #lower_solution_2 = Hungarian(g_lag_multiplier_mat * g_nb_of_dest_location + g_build_cost_dest_mat)
    t2 = datetime.datetime.now()
    for iter in range(nb_iter):
        g_sub_gradient_mat = np.zeros([g_nb_of_dest_location, g_nb_of_dest_building])
        g_assignment_mat_1 = np.zeros([g_nb_of_orig_building, g_nb_of_orig_location])
        g_assignment_mat_2 = np.zeros([g_nb_of_dest_location, g_nb_of_dest_building])
        # 【05】 GLB sub problem
        lower_solution_1 = Hungarian_1(g_GLB_cost_mat+g_build_cost_orig_mat)
        assignment_count_mat = Trace(lower_solution_1)
        for l in range(g_nb_of_orig_location):
                k_ind =lower_solution_1['location_ind'][l]
                i_ind =lower_solution_1['building_ind'][l]
                g_assignment_mat_1[i_ind][k_ind] = 1

        # 【1】 Assignment sub problem, As this is the symmetric case, we can directly use the i,k solution as the solution of this sub problem:
        lower_solution_2 = Hungarian_2(g_lag_multiplier_mat*g_nb_of_orig_location+g_build_cost_dest_mat)
        for l in range(g_nb_of_dest_location):
                l_ind =lower_solution_2['location_ind'][l]
                j_ind =lower_solution_2['building_ind'][l]
                g_assignment_mat_2[l_ind][j_ind] = 1
                g_sub_gradient_mat[l_ind][j_ind] = g_nb_of_orig_location
        print(lower_solution_1['value'])
        print(lower_solution_2['value'])


        LowerBound = lower_solution_1['value']+lower_solution_2['value']
        g_lower_bound_list.append(LowerBound)

        UpperBound = np.sum(g_flow_mat*np.matmul(np.matmul(g_assignment_mat_1,g_trans_cost_mat),g_assignment_mat_2))+np.sum(g_build_cost_orig_mat*g_assignment_mat_1)+np.sum(g_build_cost_dest_mat*g_assignment_mat_2)
        #UpperBound, g_assignment_mat_1,g_assignment_mat_2 =local_search()
        g_upper_bound_list.append(UpperBound)
        print(iter, 'iteration: Upper bound = ', UpperBound)
        print(iter, 'iteration: Lower bound = ', LowerBound)
        if LowerBound >= LB:
            LB = LowerBound
        if UpperBound <= UB:
            UB = UpperBound
            best_solution_1 = g_assignment_mat_1
            best_solution_2 = g_assignment_mat_2
        #g_lower_bound_list.append(LB)
        #g_upper_bound_list.append(UB)

        print(iter, 'iteration: Best Upper bound = ', UB)
        print(iter, 'iteration: Best Lower bound = ', LB)
        print(iter, 'iteration: GAP = ', np.abs(UB-LB)/UB)

        # [2] subgradient descent
        #print(g_sub_gradient_mat)
        #print(assignment_count_mat)
        g_sub_gradient_mat = g_sub_gradient_mat-assignment_count_mat
        g_sub_gradient_list.append(np.sum(np.abs(g_sub_gradient_mat)))
        g_lag_multiplier_list.append(g_lag_multiplier_mat)
        #print(g_lag_multiplier_mat)
        #print(g_sub_gradient_mat)
        #print(g_assignment_mat_2)
        if iter<(3*nb_iter/3):
            g_lag_multiplier_mat = g_lag_multiplier_mat+step_size*g_sub_gradient_mat
            step_size=step_size*0.5

        # elif iter>=(2*nb_iter/3):
        #     #ZZ = Cuttingplane(g_sub_gradient_list, g_lower_bound_list, g_lag_multiplier_list, UB, LB)
        #     g_lag_multiplier_mat = g_lag_multiplier_mat+step_size*g_sub_gradient_mat
        #     step_size=step_size*1
        #g_lag_multiplier_mat[g_lag_multiplier_mat<0]=0
        # print(g_branch_list[0].assignment_count)
        # print(g_branch_list[1].assignment_count)
        # print(g_branch_list[2].assignment_count)
        # print(g_branch_list[3].assignment_count)
        # print(g_sub_gradient_mat)
        # print(np.sum(np.abs(g_sub_gradient_mat)))
        #g_sub_gradient_list.append(np.sum(np.abs(g_sub_gradient_mat)))
        Branch_update()

    # ZZ = Cuttingplane(g_sub_gradient_list, g_lower_bound_list, g_lag_multiplier_list, UB, LB)
    t1 = datetime.datetime.now()
    print ('Total time usage= ', t1-t2)
    with open('output_curve.csv', 'w', newline='') as csvfile :
        writer = csv.writer(csvfile)
        writer.writerow(['lower bound', 'upper bound','subgradient'])
        for i in range(len(g_lower_bound_list)) :
            writer.writerow([g_lower_bound_list[i], g_upper_bound_list[i],g_sub_gradient_list[i]])
    best_solution_1.astype(int)
    best_solution_2.astype(int)
    np.savetxt("output_assignment_1.csv", best_solution_1, delimiter=",")
    np.savetxt("output_assignment_2.csv", best_solution_2, delimiter=",")
    print('end')
