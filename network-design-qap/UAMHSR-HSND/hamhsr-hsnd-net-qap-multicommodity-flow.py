import gurobipy
from gurobipy import *
import numpy as np
import pandas as pd
import os

# Generate distance and flow
def run_case(cwd,First_step_time,Second_step_time):
    EGLB=0 # calculate EGLB or NET-QAP (1) 0: NET-QAP (2) 1: EGLB 

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

    nb_building_1=len(building_set_1)
    nb_location_1=len(location_set_1)
    nb_building_2=len(building_set_2)
    nb_location_2=len(location_set_2)


    enviroment = gurobipy.Env()
    enviroment.setParam('TimeLimit', First_step_time)
    model=Model("quadratic_assignment",env=enviroment)

    # Create variables
    path=model.addVars(building_set_1,location_set_1,location_set_2,building_set_2,name='path',vtype=GRB.BINARY)
    assignment_1=model.addVars(building_set_1,location_set_1,name='assignment_1',vtype=GRB.BINARY)
    assignment_2=model.addVars(building_set_2,location_set_2,name='assignment_2',vtype=GRB.BINARY)

    # Assignment constraints
    for i in building_set_1:
        model.addConstr(quicksum(assignment_1[i,k] for k in location_set_1)==1,
                    "location assignment constraint[%s]%i")


    for j in building_set_2:
        model.addConstr(quicksum(assignment_2[j,l] for l in location_set_2)==1,
                    "location assignment constraint[%s]%i")

    #Coupling constraints (if we relax one of them we will obtain GLB)
    for l in location_set_2:
        for j in building_set_2:
            model.addConstr(quicksum(path[i,k,l,j] for i in building_set_1 for k in location_set_1)==nb_building_2*assignment_2[j,l],
                    "cap[%s,%s]%(l,j)")
    if EGLB==0:
        for i in building_set_1:
            for k in location_set_1:
                model.addConstr(quicksum(path[i,k,l,j] for l in location_set_2 for j in building_set_2 )==nb_building_1*assignment_1[i,k],
                        "cap[%s,%s]%(i,k)")

    # Flow conservation
    for i in building_set_1:
        for j in building_set_2:
            model.addConstr(quicksum(path[i,k,l,j] for k in location_set_1 for l in location_set_2)==1,
                            "transport_agent_sp[%s,%s]%(i,j)")



    model.setObjective(quicksum(path[i,k,l,j]*distance[k,l]*flow[i,j]
                                for i in building_set_1 for j in building_set_2 for k in location_set_1 for l in location_set_2)+\
                                    quicksum(assignment_1[i,k]*built_cost[i,k]*quicksum(flow[i,j] for j in building_set_2)                                         
                                            for i in building_set_1 for k in location_set_1)+\
                                    quicksum(assignment_2[j,l]*built_cost[l,j]*quicksum(flow[i,j] for i in building_set_1)                                         
                                            for j in building_set_2 for l in location_set_2))


    # model.Params.CrossoverBasis = 0
    # model.Params.Crossover = 0
    model.Params.BarConvTol     = 1e-3   # Basis convergence tolerance
    model.Params.OptimalityTol  = 1e-3  # Optimality tolerance
    model.Params.FeasibilityTol = 1e-3  # Feasibility tolerance

    model.Params.MIPGap = 0.5  # set MIP gap to 50% to speed up the first step
    # model.Params.NodeMethod   = 1   # Dual Simplex at nodes
    # model.Params.NumericFocus = 2
    # model.Params.ScaleFlag    = 1
    # model.Params.Presolve     = 2
    model.optimize()

    # warm-start-cputime: 360s
    net_qap_obj=model.ObjVal
    net_qap_time=model.Runtime
    net_qap_gap=model.MIPGap
    print("warm-start results:")
    open_performance = os.path.join(cwd,'output_performance(hybrid-multicommodity-flow)-step1.csv')
    with open(open_performance, 'w') as f:
        # objective value, best_lower_bound, computation time, gap
        f.write(f"Objective Value,{model.ObjVal}\n")
        f.write(f"Best Lower Bound,{model.ObjBound}\n")
        f.write(f"Computation Time,{model.Runtime}\n")
        f.write(f"Gap,{model.MIPGap}\n")
    print(f"Objective_value:{net_qap_obj},Computation_time:{net_qap_time},Gap:{net_qap_gap}")
    second_step_flag = 0
    print("==============================================================")
    # # =======================
    # # Run Multicommodity Flow Model if GAP > 1%
    # Use the solution from the previous model as a warm start
    if model.MIPGap>0.01:
        second_step_flag = 1
        assignment_solution_1={}
        assignment_solution_2={}
        for i in building_set_1:
            for k in location_set_1:
                assignment_solution_1[i,k]=assignment_1[i,k].X
        for j in building_set_2:
            for l in location_set_2:
                assignment_solution_2[j,l]=assignment_2[j,l].X
        # use the previous solution as a warm start
        best_obj=model.ObjVal
        best_lb=model.ObjBound


        # create a new model
        enviroment = gurobipy.Env()
        enviroment.setParam('TimeLimit', Second_step_time)
        model=Model("quadratic_assignment",env=enviroment)
        # Create variables
        flow_1=model.addVars(building_set_1,building_set_1,location_set_1,building_set_2,name='first_stage')
        flow_2=model.addVars(building_set_1,location_set_2,building_set_2,building_set_2,name='second_stage')
        flow_t=model.addVars(building_set_1,location_set_1,location_set_2,building_set_2,name='transshipment')
        assignment_1=model.addVars(building_set_1,location_set_1,name='assignment_1',vtype=GRB.BINARY)
        assignment_2=model.addVars(building_set_2,location_set_2,name='assignment_2',vtype=GRB.BINARY)
        # use the previous solution as a warm start
        for i in building_set_1:
            for k in location_set_1:
                assignment_1[i,k].Start=assignment_solution_1[i,k]
        for j in building_set_2:
            for l in location_set_2:
                assignment_2[j,l].Start=assignment_solution_2[j,l]
        
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

    


        # add the best lower bound from the previous model as a cut
        obj_expr = quicksum(quicksum(flow_t[i,k,l,j] for i in building_set_1 for j in building_set_2)*distance[k,l] for k in location_set_1 for l in location_set_2)+\
                                        quicksum(assignment_1[i,k]*built_cost[i,k]*quicksum(flow[i,j] for j in building_set_2)                                         
                                                for i in building_set_1 for k in location_set_1)+\
                                        quicksum(assignment_2[j,l]*built_cost[l,j]*quicksum(flow[i,j] for i in building_set_1)                                         
                                                for j in building_set_2 for l in location_set_2)
        
        model.setObjective(obj_expr)
        # model.addConstr(obj_expr>=best_lb,"best_lb_cut")
        # model.addConstr(obj_expr<=best_obj,"best_obj_cut")


        model.Params.BarConvTol     = 1e-3   # Basis convergence tolerance
        model.Params.OptimalityTol  = 1e-3  # Optimality tolerance
        model.Params.FeasibilityTol = 1e-3  # Feasibility tolerance

        model.Params.Method = 1   # Dual Simplex as the initial method
        model.Params.NodeMethod   = 1   # Dual Simplex at nodes
        model.Params.Cutoff = best_obj - 1e-6 # set cutoff to best obj - small epsilon
        model.Params.MIPGap = 1e-4  # set MIP gap to 0.01%
        model.optimize()


    if model.SolCount == 0:
        print("No feasible solution found")
    else:
        # open a file for writing
        open_path = os.path.join(cwd,'output_path_solution(hybrid-multicommodity-flow).csv')
        open_assignment_1 = os.path.join(cwd,'output_assignment_solution_1(hybrid-multicommodity-flow).csv')
        open_assignment_2 = os.path.join(cwd,'output_assignment_solution_2(hybrid-multicommodity-flow).csv')
        open_performance = os.path.join(cwd,'output_performance(hybrid-multicommodity-flow).csv')
        if second_step_flag==0:
            path_solution = model.getAttr('x', path)
        else:
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
        total_time=net_qap_time+model.Runtime
        print(f"Objective_value:{model.ObjVal},best_lower_bound:{model.ObjBound},Computation_time:{total_time},Gap:{model.MIPGap}")
    
if __name__ == "__main__":
    # Generate distance and flow
    case_list = [
    # [r'UAMHSR-HSND cases\\case40-6-6-40', 1800, 1800],
    # [r'UAMHSR-HSND cases\\case40-50-50-40', 1800, 1800],
    # [r'UAMHSR-HSND cases\\case50-7-7-50', 1800, 1800],
    # [r'UAMHSR-HSND cases\\case50-8-8-50', 1800, 1800],
    # [r'UAMHSR-HSND cases\\case50-50-50-50', 1800, 1800],
    # [r'UAMHSR-HSND cases\\case60-50-50-60', 1800, 1800],
    [r'UAMHSR-HSND cases\\case70-50-50-70', 1800, 1800]
    ]

    for case in case_list:
        cwd = case[0]
        First_step_time = case[1]
        Second_step_time = case[2]
        print("**********===== Running case: ", cwd, " using hybrid multicommodity-flow model ============************")
        run_case(cwd,First_step_time,Second_step_time)




