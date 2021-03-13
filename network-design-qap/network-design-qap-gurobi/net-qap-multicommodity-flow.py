from gurobipy import *
import numpy as np
import pandas as pd

# Generate distance and flow
cwd = r'..\NEOs_6'

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
enviroment.setParam('TimeLimit', 360)
model=Model("quadratic_assignment",env=enviroment)

# Create variables
flow_1=model.addVars(building_set_1,building_set_1,location_set_1,building_set_2,name='first_stage')
flow_2=model.addVars(building_set_1,location_set_2,building_set_2,building_set_2,name='second_stage')
flow_t=model.addVars(building_set_1,location_set_1,location_set_2,building_set_2,name='transshipment')

assignment_1=model.addVars(building_set_1,location_set_1,name='assignment_1',vtype=GRB.BINARY)
assignment_2=model.addVars(building_set_2,location_set_2,name='assignment_2',vtype=GRB.BINARY)

# # Assignment constraints
for k in location_set_1:
    model.addConstr(quicksum(assignment_1[i,k] for i in building_set_1)==1,
                   "building assignment constraint[%s]%k")
for i in building_set_1:
    model.addConstr(quicksum(assignment_1[i,k] for k in location_set_1)==1,
                   "location assignment constraint[%s]%i")

for l in location_set_2:
    model.addConstr(quicksum(assignment_2[j,l] for j in building_set_2)==1,
                   "building assignment constraint[%s]%k")

for j in building_set_2:
    model.addConstr(quicksum(assignment_2[j,l] for l in location_set_2)==1,
                   "location assignment constraint[%s]%i")

# #Capacity-constraints
M=100000000
for l in location_set_2:
    for j in building_set_2:
        model.addConstr(quicksum(flow_2[i,l,j,j] for i in building_set_1)<=M*assignment_2[j,l],
                   "cap[%s,%s]%(l,j)")

for i in building_set_1:
    for k in location_set_1:
        model.addConstr(quicksum(flow_1[i,i,k,j] for j in building_set_2 )<=M*assignment_1[i,k],
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
                                quicksum(assignment_1[i,k]*built_cost[i,k] for i in building_set_1 for k in location_set_1)+\
                                quicksum(assignment_2[j,l]*built_cost[l,j] for j in building_set_2 for l in location_set_2))

model.optimize()
print(1)
path_solution = model.getAttr('x', path)
for key in path_solution.keys():
    if path_solution[key]>0:
        print(key, path_solution[key])

solution_1 = model.getAttr('x', assignment_1)
solution_2 = model.getAttr('x', assignment_2)
assignment_s_1=np.matrix([[solution_1[i,k] for k in location_set_1] for i in building_set_1])
print(assignment_s_1)

assignment_s_2=np.matrix([[solution_2[j,l] for l in location_set_2] for j in building_set_2])
print(assignment_s_2)