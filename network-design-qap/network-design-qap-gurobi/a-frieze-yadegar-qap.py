from gurobipy import *
import numpy as np
import pandas as pd
import os 
R_A_FY_QAP=0 # RDD lower bound or original FY model (1) 1: RDD lowerbound, 0: A-FY-QAP
cwd = r'..\NEOS_6'

link_df=pd.read_csv(os.path.join(cwd,'input_link.csv'))
agent_df=pd.read_csv(os.path.join(cwd,'input_agent.csv'))
node_df=pd.read_csv(os.path.join(cwd,'input_node.csv'))

agent_df['od_pair']=agent_df.apply(lambda x: (x.origin_node_id,x.destination_node_id),axis=1)
flow=agent_df[['od_pair','customized_cost_link_value']].set_index('od_pair').to_dict()['customized_cost_link_value']
link_df['od_pair']=link_df.apply(lambda x: (x.from_node_id,x.to_node_id),axis=1)
distance=link_df[['od_pair','trans_cost']].set_index('od_pair').to_dict()['trans_cost']

built_cost=link_df[['od_pair','built_cost']].set_index('od_pair').to_dict()['built_cost']

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

enviroment = gurobipy.Env()
enviroment.setParam('TimeLimit', 360)
model=Model("quadratic_assignment",env=enviroment)

# Create variables

if R_A_FY_QAP==1:
    assignment_1=model.addVars(building_set_1,location_set_1,name='assignment_1',lb=0,ub=1)
    assignment_2=model.addVars(building_set_2,location_set_2,name='assignment_2',lb=0,ub=1)
    path=model.addVars(building_set_1,location_set_1,location_set_2,building_set_2,name='path',lb=0,ub=1)
elif R_A_FY_QAP==0:
    assignment_1=model.addVars(building_set_1,location_set_1,name='assignment_1',vtype=GRB.BINARY)
    assignment_2=model.addVars(building_set_2,location_set_2,name='assignment_2',vtype=GRB.BINARY)
    path=model.addVars(building_set_1,location_set_1,location_set_2,building_set_2,name='path',lb=0,ub=1)

# Assignment constraints
for k in location_set_1:
    model.addConstr(quicksum(assignment_1[i,k] for i in building_set_1)==1,
                   "building assignment constraint[%s]%k")
for i in building_set_1:
    model.addConstr(quicksum(assignment_1[i,k] for k in location_set_1)==1,
                   "location assignment constraint[%s]%i")
# Assignment constraints
for l in location_set_2:
    model.addConstr(quicksum(assignment_2[j,l] for j in building_set_2)==1,
                   "building assignment constraint[%s]%k")
for j in building_set_2:
    model.addConstr(quicksum(assignment_2[j,l] for l in location_set_2)==1,
                   "location assignment constraint[%s]%i")

# capacity constraints
##Relax the following two constraints when calculate GLB
for k in location_set_1:
    for l in location_set_2:
        for j in building_set_2:
            model.addConstr(quicksum(path[i,k,l,j] for i in building_set_1)==assignment_2[j,l],
                   "cap[%s,%s,%s]%(k,l,j)")

for i in building_set_1:
    for l in location_set_2:
        for j in building_set_2:
            model.addConstr(quicksum(path[i,k,l,j] for k in location_set_1)==assignment_2[j,l],
                   "cap[%s,%s,%s]%(i,l,j)")


for i in building_set_1:
    for k in location_set_1:
        for l in location_set_2:
            model.addConstr(quicksum(path[i,k,l,j] for j in building_set_2)==assignment_1[i,k],
                   "cap[%s,%s,%s]%(i,k,l)")

for i in building_set_1:
    for k in location_set_1:
        for j in building_set_2:
            model.addConstr(quicksum(path[i,k,l,j] for l in location_set_2)==assignment_1[i,k],
                   "cap[%s,%s,%s]%(i,k,j)")

#model.addConstr(quicksum(path[i,k,l,j]*distance[k,l]*flow[i,j] for i in building_set_1 for j in building_set_2 for k in location_set_1 for l in location_set_2)>=394)


model.setObjective(quicksum(path[i,k,l,j]*distance[k,l]*flow[i,j]
                            for i in building_set_1 for j in building_set_2 for k in location_set_1 for l in location_set_2)+\
                                quicksum(assignment_1[i,k]*built_cost[i,k] for i in building_set_1 for k in location_set_1)+\
                                quicksum(assignment_2[j,l]*built_cost[l,j] for j in building_set_2 for l in location_set_2))
model.optimize()
print(model.getAttr('x',assignment_1))
print(model.getAttr('x',assignment_2))