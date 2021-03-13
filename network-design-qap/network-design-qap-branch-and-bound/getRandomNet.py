# -*- coding:utf-8 -*-
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.rc('font',size=9.5)
import random
import os
cwd = r'..\large_scale_synchronization_r4'

import csv

def main(nb_DC,nb_warehouse,nb_pick_up_station,square_length):
    with open(os.path.join(cwd,'node.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node_name', 'node_id','x_coord', 'y_coord','FT'])
        node_id=1
        print('distribution center')   
        count = 0
        x_dc = np.zeros(nb_DC)
        y_dc = np.zeros(nb_DC)
        while count < nb_DC:
            x_dc[count] = np.random.randint(-square_length*0.9,square_length*0.9)
            y_dc[count] = np.random.randint(-square_length*0.9,square_length*0.9)
            print(x_dc[count], y_dc[count])
            line=['DC'+str(count+1),node_id,x_dc[count],y_dc[count],'distribution_center' ]
            writer.writerow(line)
            node_id+=1
            count += 1
        print('random generate '+str(count)+' distribution centers...')

        transshipment_cost=np.zeros([nb_DC*nb_rec,nb_DC*nb_del])
        for k in range(nb_rec):
            for l in range(nb_del):
                for i in range(nb_DC):
                    for j in range(nb_DC):
                        # mahanttan distance 
                        transshipment_cost[k*nb_DC+i][l*nb_DC+j]=abs(y_dc[i]-y_dc[j])+abs(x_dc[i]-x_dc[j])
    

        # convert array into dataframe 
        DF = pd.DataFrame(transshipment_cost)   
        # save the dataframe as a csv file 
        DF.to_csv(os.path.join(cwd,"transshipment_time.csv"),index=False,header=False)    

        print('warehouse')    
        x_w = np.zeros(nb_warehouse)
        y_w = np.zeros(nb_warehouse)  
        count = 0
        while count < nb_warehouse:
            x_w[count] = np.random.randint(-square_length,square_length)
            y_w[count] = np.random.randint(-square_length,square_length)

            print(x_w[count], y_w[count])
            line=['WH'+str(count+1),node_id,x_w[count],y_w[count],'warehouse']
            writer.writerow(line)
            node_id+=1            
            count += 1
        print('random generate '+str(count)+' warehouses...')

        travel_time_1=np.zeros([nb_DC*nb_rec,nb_DC*nb_rec])
        for k in range(nb_rec):
            for i in range(nb_warehouse):
                for j in range(nb_DC):
                    # mahanttan distance 
                    travel_time_1[i][k*nb_DC+j]=(abs(y_w[i]-y_dc[j])+abs(x_w[i]-x_dc[j]))/2
        # convert array into dataframe 
        DF = pd.DataFrame(travel_time_1)   
        # save the dataframe as a csv file 
        DF.to_csv(os.path.join(cwd,"travel_time_1.csv"),index=False,header=False)         

        print('pick-up station')    
        x_s = np.zeros(nb_pick_up_station)
        y_s = np.zeros(nb_pick_up_station)  
        count = 0
        while count < nb_pick_up_station:
            x_s[count] = np.random.randint(-square_length,square_length)
            y_s[count] = np.random.randint(-square_length,square_length)

            print(x_s[count], y_s[count])
            line=['PS'+str(count+1),node_id,x_s[count],y_s[count],'pick-up_station']
            writer.writerow(line)
            node_id+=1
            count += 1
        print('random generate '+str(count)+' pick up stations...')

        travel_time_2=np.zeros([nb_DC*nb_del,nb_DC*nb_del])
        for k in range (nb_del):
            for i in range(nb_pick_up_station):
                for j in range(nb_DC):
                    travel_time_2[nb_DC*k+j][i]=(abs(y_s[i]-y_dc[j])+abs(x_s[i]-x_dc[j]))/2

    # convert array into dataframe 
    DF = pd.DataFrame(travel_time_2)   
    # save the dataframe as a csv file 
    DF.to_csv(os.path.join(cwd,"travel_time_2.csv"),index=False,header=False)

    plt.figure(figsize=(8,8.1),dpi=125)
    plt.plot(x_dc,y_dc,'o',label='Distribution centers',markersize=8,c='k')
    plt.plot(x_w,y_w,'D', label ='Warehouses',markersize=5,c='b')
    plt.plot(x_s,y_s,'D', label='Pick-up stations',markersize=5,c='r')
    
    plt.xlim((-square_length-3,square_length+3))   
    plt.ylim((-square_length-3,square_length+3))

    my_x_ticks = np.arange(-square_length-3,square_length+3, 1)
    my_y_ticks = np.arange(-square_length-3,square_length+3, 1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Random Scatter')
    plt.savefig(os.path.join(cwd,'imag.png'))
    plt.show()
    return travel_time_1,travel_time_2,transshipment_cost

if __name__=='__main__':
    receive_wave=[11,16]
    order_time=[6,9,10,14,17,6,9,11,14,17,7,10,12,15,8,16]
    delivery_wave=[6,14]
    pick_up_time=[8,9.5,16,18,7.5,10,17,19,8,9.5,17,19,20,21,17.5,13,8,9,11,15]
    period_time=24
    timewindow=24

    global nb_rec
    global nb_del
    
    nb_rec=len(receive_wave)
    nb_del=len(delivery_wave)
    nb_DC=12
    nb_warehouse=len(order_time)
    nb_station=len(pick_up_time)
    city_radius=6
    fix_transport_cost=2 # dollar per hour 
    fix_inventory_cost=0.1 # dollar per hour 
    var_transport_rate=1 # dollar per hour 
    var_inventory_cost=0.1 # dollar per hour 

    data_flow = np.loadtxt(os.path.join(cwd,'t_flow_matrix.txt'))
    print('first-stage timetable...')
    timetable_1=np.zeros((2,nb_DC*nb_rec))
    for i in range(2):
        timetable_1[0][i*nb_DC:i*nb_DC+nb_DC]=receive_wave[i]
        timetable_1[1,:]=-1
    for w in range(nb_warehouse):
        timetable_1[1,w]=np.random.randint(0,24)
        timetable_1[1,w]=order_time[w]
    timetable_o=timetable_1
    # convert array into dataframe 
    DF = pd.DataFrame(timetable_1)   
    # save the dataframe as a csv file 
    DF.to_csv(os.path.join(cwd,"timetable_1.csv"),index=False,header=False)

    print('second-stage timetable...')
    timetable_2=np.zeros((2,nb_DC*nb_del))
    for i in range(2):
        timetable_2[1][i*nb_DC:i*nb_DC+nb_DC]=receive_wave[i]
        timetable_2[0,:]=-1
    for s in range(nb_station):
        timetable_2[0,s]=np.random.randint(0,24)
        timetable_2[0,s]=pick_up_time[s]
    timetable_d=timetable_2

    # convert array into dataframe 
    DF = pd.DataFrame(timetable_2)   
    # save the dataframe as a csv file 
    DF.to_csv(os.path.join(cwd,"timetable_2.csv"),index=False,header=False)

    travel_time_o,travel_time_d,transshipment_time=main(nb_DC,nb_warehouse,nb_station,city_radius)


    q,n= timetable_o.shape
    q,m= timetable_d.shape
    data_built_1=np.zeros([n,n])
    data_built_2=np.zeros([m,m])
    data_dis=np.zeros([n,m])
    for i in range(n):
        if timetable_o[1][i] !=-1:
            for k in range(n):
                tmp=np.mod(timetable_o[1][i]+travel_time_o[i][k],period_time)
                nb_of_period=abs(np.floor((timetable_o[1][i]+travel_time_o[i][k])/period_time))
                if timetable_o[0][k]<tmp:
                    data_built_1[i][k]=period_time-timetable_o[1][i]+timetable_o[0][k]+nb_of_period*period_time
                if timetable_o[0][k]>=tmp:
                    data_built_1[i][k]=timetable_o[0][k]-timetable_o[1][i]+nb_of_period*period_time
        elif timetable_o[1][i] ==-1:
            for k in range(n):
                data_built_1[i][k]=0
    
    

    for j in range(m):         
        if timetable_d[0][j] !=-1:
            for l in range(m):
                tmp=np.mod(timetable_d[0][j]-travel_time_d[l][j],period_time)
                nb_of_period=abs(np.floor((timetable_o[1][i]+travel_time_o[i][k])/period_time))
                if timetable_d[1][l]>tmp:
                    data_built_2[l][j]=period_time-timetable_d[1][l]+timetable_d[0][j]
                if timetable_d[1][l]<=tmp:
                    data_built_2[l][j]=timetable_d[0][j]-timetable_d[1][l]
        elif timetable_d[0][j] ==-1:
            for l in range(m):
                data_built_2[l][j]=0

    for i in range(n):
        for k in range(n):
            if (travel_time_o[i][k]+timewindow<data_built_1[i][k]):
                if travel_time_o[i][k] !=0:
                    data_built_1[i][k]=100000000

    for l in range(m):
        for j in range(m):
            if (travel_time_d[l][j]+timewindow<data_built_2[l][j]):
                if travel_time_d[l][j] !=0:
                    data_built_2[l][j]=100000000


    data_built_1_A=(data_built_1-travel_time_o)*fix_inventory_cost+travel_time_o*fix_transport_cost
    data_built_2_A=(data_built_2-travel_time_d)*fix_inventory_cost+travel_time_d*fix_transport_cost

    for k in range(n):
        for l in range(m):
            tmp=np.mod(timetable_o[0][k]+transshipment_time[k][l],period_time)
            nb_of_period=abs(np.floor((timetable_o[1][i]+travel_time_o[i][k])/period_time))
            if tmp<timetable_d[1][l]:
               data_dis[k][l] = (timetable_d[1][l]-timetable_o[0][k])+nb_of_period*period_time

            if tmp>=timetable_d[1][l]:
                data_dis[k][l]=(period_time-timetable_o[0][k]+timetable_d[1][l])+nb_of_period*period_time

    data_dis=(data_dis-transshipment_time)*var_inventory_cost+transshipment_time*var_transport_rate




    built1_df=pd.DataFrame(data_built_1)
    built2_df=pd.DataFrame(data_built_2)
    flow_df=pd.DataFrame(data_flow)

    transshipment_df=pd.DataFrame(data_dis)
    built1_df.to_csv(os.path.join(cwd,'built1.csv'))
    built2_df.to_csv(os.path.join(cwd,'built2.csv'))
    transshipment_df.to_csv(os.path.join(cwd,'transshipment_cost.csv'))
    flow_df.to_csv(os.path.join(cwd,'flow.csv'))

    n,m=data_flow.shape
    node_num_each = [n,n,m,m]
    #Read Data
    nodename_array=['building node1','location node1','location node2','building node2']





    with open(os.path.join(cwd,'input_node.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node_name', 'node_id','x', 'y'])
        for i in  range(0,4):
            for j in range(node_num_each[i]):
                if(i<4):
                    nodeid=1000*(i+1)+j+1
                    locationx = 100 * i
                    locationy = 10 * j
                line = [nodename_array[i],
                        nodeid,
                        locationx,
                        locationy]
                writer.writerow(line)



    with open(os.path.join(cwd,'input_link.csv'), 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            ['link_id','from_node_id', 'to_node_id', 'built_cost','trans_cost','link_type','acting_link_id'])
        count=0
        for i in range(node_num_each[0]):# building1 to location1
            for j in range(node_num_each[1]):
                count+=1
                line1=[count,
                    1000+i+1,
                    2000+j+1,
                    data_built_1[i][j], # built cost
                    0, #trans cost
                    2,#building to location
                    count]
                writer.writerow(line1)
        for i in range(node_num_each[2]):# location 2 to building 2
            for j in range(node_num_each[3]):
                count+=1
                line2=[count,
                    3000+i+1,
                    4000+j+1,
                    data_built_2[i][j], # built cost
                    0,# trans cost
                    3,#location to building
                    count]
                writer.writerow(line2)
                # print('node',node_num_each)
                # print('x',i)
                # print('y',j)
                # print(data_built_2[i][j])
        for i in range(node_num_each[1]): # location 1 to location 2 transportation
            for j in range(node_num_each[2]):
                #if(j!=i):
                count += 1
                line3 = [count,
                    2000 + i + 1,
                    3000 + j + 1,
                    0, # built
                    data_dis[i][j],# trans
                    1,#physical transportation link
                    count]
                writer.writerow(line3)


    with open(os.path.join(cwd,'input_agent.csv'), 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            ['agent_id','origin_node_id','destination_node_id','customized_cost_link_type',
            'customized_cost_link_value','agent_type','set_of_allowed_link_types'])
        count = 0
        # transportation agent
        for i in range(node_num_each[0]):
            for j in range(node_num_each[3]):
                count += 1
                line1 = [count,
                        1000 + i + 1,
                        4000 + j + 1,
                        1,
                        data_flow[i][j],#customized_cost:the flow(i,j)
                        1,
                        '1;2;3;4']
                writer.writerow(line1)
                    #print(data_flow[i][j])
