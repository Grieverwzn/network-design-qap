import gurobipy
print(gurobipy.gurobi.version())
from gurobipy import *
import numpy as np
import pandas as pd
import os


def run_case(cwd,time_limit):
    link_df=pd.read_csv(os.path.join(cwd,'input_link.csv'))
    agent_df=pd.read_csv(os.path.join(cwd,'input_agent.csv'))
    node_df=pd.read_csv(os.path.join(cwd,'input_node.csv'))


    agent_df['od_pair']=agent_df.apply(lambda x: (x.origin_node_id,x.destination_node_id),axis=1)
    flow=agent_df[['od_pair','customized_cost_link_value']].set_index('od_pair').to_dict()['customized_cost_link_value']
    link_df['od_pair']=link_df.apply(lambda x: (x.from_node_id,x.to_node_id),axis=1)
    distance=link_df[['od_pair','trans_cost']].set_index('od_pair').to_dict()['trans_cost']
    built_cost=link_df[['od_pair','built_cost']].set_index('od_pair').to_dict()['built_cost']
    # Build up the set of building and locations
    building_set_1=[]
    building_set_2=[]
    location_set_1=[]
    location_set_2=[]
    building_set=[]
    location_set=[]
    building_set_map=[]
    location_set_map=[]
    node_set=[]

    for i in range(len(node_df)):
        if node_df.iloc[i].node_name == 'building node1':
            building_set_1.append(node_df.iloc[i].node_id)
        if node_df.iloc[i].node_name == 'building node2':
            building_set_2.append(node_df.iloc[i].node_id)
        if node_df.iloc[i].node_name == 'location node1':
            location_set_1.append(node_df.iloc[i].node_id)
        if node_df.iloc[i].node_name == 'location node2':
            location_set_2.append(node_df.iloc[i].node_id) 


    location_set.extend(location_set_1)
    location_set.extend(location_set_2)
    location_set_map.extend(location_set_2)
    location_set_map.extend(location_set_1)

    building_set.extend(building_set_1)
    building_set.extend(building_set_2)
    building_set_map.extend(building_set_2)
    building_set_map.extend(building_set_1)

    node_set=location_set
    node_set.extend(building_set_map)
    nb_building_1=len(building_set_1)
    nb_location_1=len(location_set_1)
    nb_building_2=len(building_set_2)
    nb_location_2=len(location_set_2)




    enviroment = gurobipy.Env()
    enviroment.setParam('TimeLimit', time_limit)
    model=Model("quadratic_assignment",env=enviroment)

    # Create variables
    flow_1=model.addVars(building_set_1,building_set_1,location_set_1,building_set_2,name='first_stage')
    flow_2=model.addVars(building_set_1,location_set_2,building_set_2,building_set_2,name='second_stage')
    flow_t=model.addVars(building_set_1,location_set_1,location_set_2,building_set_2,name='transshipment')

    assignment_1=model.addVars(building_set_1,location_set_1,name='assignment_1',vtype=GRB.BINARY)
    assignment_2=model.addVars(building_set_2,location_set_2,name='assignment_2',vtype=GRB.BINARY)

    # # Assignment constraints
    for i in building_set_1:
        model.addConstr(quicksum(assignment_1[i,k] for k in location_set_1)==1,
                    "location assignment constraint[%s]%i")

    for j in building_set_2:
        model.addConstr(quicksum(assignment_2[j,l] for l in location_set_2)==1,
                    "location assignment constraint[%s]%i")

    # #Capacity-constraints
    M_i = {i: sum(flow[i,j] for j in building_set_2) for i in building_set_1}
    M_j = {j: sum(flow[i,j] for i in building_set_1) for j in building_set_2}

    for l in location_set_2:
        for j in building_set_2:
            model.addConstr(quicksum(flow_2[i,l,j,j] for i in building_set_1)<=M_j[j]*assignment_2[j,l],
                    "cap[%s,%s]%(l,j)")

    for i in building_set_1:
        for k in location_set_1:
            model.addConstr(quicksum(flow_1[i,i,k,j] for j in building_set_2 )<=M_i[i]*assignment_1[i,k],
                    "cap[%s,%s]%(i,k)")

    # Flow conservation constraints 
    for i in building_set_1:
        for j in building_set_2:
            for i in building_set_1:
                model.addConstr(-quicksum(flow_1[i,i,k,j] for k in location_set_1)==-flow[i,j],
                            "first_stage_flow_balance[%s,%s]%(i,j)")

    for i in building_set_1:
        for j in building_set_2:
            for j in building_set_2:
                model.addConstr(quicksum(flow_2[i,l,j,j] for l in location_set_2)==flow[i,j],
                            "second_stage_flow_balance[%s,%s]%(i,j)")

    for i in building_set_1:
        for j in building_set_2:
            for k in location_set_1:
                model.addConstr(flow_1[i,i,k,j]-quicksum(flow_t[i,k,l,j] for l in location_set_2)==0,
                            "second_stage_flow_balance[%s,%s]%(i,j)")

    for i in building_set_1:
        for j in building_set_2:
            for l in location_set_2:
                model.addConstr(quicksum(flow_t[i,k,l,j] for k in location_set_1)-flow_2[i,l,j,j]==0,
                            "second_stage_flow_balance[%s,%s]%(i,j)")


    model.setObjective(quicksum(quicksum(flow_t[i,k,l,j] for i in building_set_1 for j in building_set_2)*distance[k,l] for k in location_set_1 for l in location_set_2)+\
                                    quicksum(assignment_1[i,k]*built_cost[i,k]*quicksum(flow[i,j] for j in building_set_2)                                         
                                            for i in building_set_1 for k in location_set_1)+\
                                    quicksum(assignment_2[j,l]*built_cost[l,j]*quicksum(flow[i,j] for i in building_set_1)                                         
                                            for j in building_set_2 for l in location_set_2))


    # model.Params.CrossoverBasis = 0
    # model.Params.Crossover = 0
    model.Params.BarConvTol     = 1e-3   # Basis convergence tolerance
    model.Params.OptimalityTol  = 1e-3  # Optimality tolerance
    model.Params.FeasibilityTol = 1e-3  # Feasibility tolerance


    # model.Params.NodeMethod   = 1   # Dual Simplex at nodes
    # model.Params.NumericFocus = 2
    # model.Params.ScaleFlag    = 1
    # model.Params.Presolve     = 2
    model.optimize()


    if model.SolCount == 0:
        print("No feasible solution found")
    else:
        # open a file for writing
        open_path = os.path.join(cwd,'output_path_solution(multicommodity-flow).csv')
        open_assignment_1 = os.path.join(cwd,'output_assignment_solution_1(multicommodity-flow).csv')
        open_assignment_2 = os.path.join(cwd,'output_assignment_solution_2(multicommodity-flow).csv')
        open_performance = os.path.join(cwd,'output_performance(multicommodity-flow).csv')

        path_solution = model.getAttr('x', flow_t)
        with open(open_path, 'w') as f:
            f.write('building_1,location_1,location_2,building_2,assignment\n')
            for key in path_solution.keys():
                if path_solution[key]>0:
                    f.write(f"{key[0]},{key[1]},{key[2]},{key[3]},{path_solution[key]}\n")
        assignment_solution_1 = model.getAttr('x', assignment_1)
        with open(open_assignment_1, 'w') as f:
            f.write('building_1,location_1,assignment\n')
            for key in assignment_solution_1.keys():
                if assignment_solution_1[key]>0:
                    f.write(f"{key[0]},{key[1]},{assignment_solution_1[key]}\n")
        assignment_solution_2 = model.getAttr('x', assignment_2)
        with open(open_assignment_2, 'w') as f:
            f.write('building_2,location_2,assignment\n')
            for key in assignment_solution_2.keys():
                if assignment_solution_2[key]>0:
                    f.write(f"{key[0]},{key[1]},{assignment_solution_2[key]}\n")
        
        with open(open_performance, 'w') as f:
            # objective value, best_lower_bound, computation time, gap
            f.write(f"Objective Value,{model.ObjVal}\n")
            f.write(f"Best Lower Bound,{model.ObjBound}\n")
            f.write(f"Computation Time,{model.Runtime}\n")
            f.write(f"Gap,{model.MIPGap}\n")
        # close the file
        f.close()
        total_time=model.Runtime
        print(f"Objective_value:{model.ObjVal},best_lower_bound:{model.ObjBound},Computation_time:{total_time},Gap:{model.MIPGap}")
    
    
if __name__ == "__main__":
    # Generate distance and flow
    case_list = [
    (r'UAMHSR-HSND cases\\case6-3-3-6', 360),
    (r'UAMHSR-HSND cases\\case6-6-6-6', 360),
    (r'UAMHSR-HSND cases\\case7-7-7-7', 360),
    (r'UAMHSR-HSND cases\\case8-8-8-8', 360),
    (r'UAMHSR-HSND cases\\case9-9-9-9', 360),
    (r'UAMHSR-HSND cases\\case10-10-10-10', 360),
    (r'UAMHSR-HSND cases\\case20-3-3-20', 360),
    (r'UAMHSR-HSND cases\\case20-4-4-20', 360),
    (r'UAMHSR-HSND cases\\case30-5-5-30', 360),
    (r'UAMHSR-HSND cases\\case40-6-6-40', 360),
    (r'UAMHSR-HSND cases\\case30-50-50-30', 360),
    (r'UAMHSR-HSND cases\\case40-50-50-40', 360),
    (r'UAMHSR-HSND cases\\case50-7-7-50', 1800),
    (r'UAMHSR-HSND cases\\case50-8-8-50', 1800),
    (r'UAMHSR-HSND cases\\case50-50-50-50', 1800),
    (r'UAMHSR-HSND cases\\case60-50-50-60', 1800),
    (r'UAMHSR-HSND cases\\case70-50-50-70', 1800)]

    for case in case_list:
        cwd = case[0]
        time_limit = case[1]
        print("**********===== Running case: ", cwd, " using multicommodity-flow model ============************")
        run_case(cwd,time_limit)
    