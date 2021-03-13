import pandas as pd
import os
import time
import numpy as np


class Node:
    def __init__(self):
        self.name = ''
        self.node_id = 0
        self.m_outgoing_link_list = []
        self.m_incoming_link_list = []


class Link:
    def __init__(self):
        self.link_id = 0
        self.link_type = 0
        self.from_node_id = 0
        self.to_node_id = 0
        self.from_node = None
        self.to_node = None
        self.built_cost = 0.0
        self.trans_cost = 0.0
        self.multiplier = 0.0


class Agent:
    def __init__(self):
        self.agent_id = 0
        self.agent_type = 0
        # self.agent_list_seq_no = 0
        self.origin_node_id = 0
        self.destination_node_id = 0
        self.origin_node = None
        self.destination_node = None
        self.customized_cost_link_type = 0
        self.flow = 0.0
        self.set_of_allowed_links_types = []
        self.path_cost = 0
        self.path_node_seq_no_list = []
        self.path_link_seq_no_list = []
        self.path_node_seq_str = ''
        self.path_time_seq_str = ''
        self.number_of_nodes = 0
        self.path_cost = 0
        self.m_path_link_seq_no_list_size = 0
        self.m_current_link_seq_no = 0


class Branch:  # it is the branch for Gilmore Lawler bound
    def __init__(self, from_node_id, to_node_id):
        self.branch_id = 0
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.i_ind = 0
        self.k_ind = 0

        self.trans_cost_array = None
        self.flow_array = None


class ParamsQAP:
    def __init__(self, cwd='', args=None):
        self.cwd = cwd
        self.args = args
        self.nb_of_nodes = 0
        self.nb_of_links = 0
        self.nb_of_agents = 0
        self.node_list = []
        self.link_list = []
        self.agent_list = []
        self.node_id_to_node_dict = {}

        self.nb_of_orig_building = 0
        self.nb_of_orig_location = 0
        self.nb_of_dest_building = 0
        self.nb_of_dest_location = 0
        self.building_orig_list = []
        self.building_dest_list = []
        self.location_orig_list = []
        self.location_dest_list = []

        self.flow_mat = None
        self.trans_cost_mat = None
        self.build_cost_orig_mat = None
        self.build_cost_dest_mat = None
        self.GLB_cost_mat = None

        self.nb_of_branch = 0
        self.branch_list = []

        self.readInputData()
        self.initialization()

    def readInputData(self):
        print('Start input data...')
        t0 = time.time()

        node_df = pd.read_csv(os.path.join(self.cwd,'input_node.csv'), encoding='gbk')
        self.nb_of_nodes = len(node_df)
        for i in range(self.nb_of_nodes):
            node = Node()
            node.node_id = node_df.loc[i,'node_id']
            node.name = node_df.loc[i,'node_name']
            self.node_list.append(node)

            self.node_id_to_node_dict[node.node_id] = node
            if node.name == 'building node1':
                self.building_orig_list.append(node)
            elif node.name == 'building node2':
                self.building_dest_list.append(node)
            elif node.name == 'location node1':
                self.location_orig_list.append(node)
            elif node.name == 'location node2':
                self.location_dest_list.append(node)


        link_df = pd.read_csv(os.path.join(self.cwd,'input_link.csv'), encoding='gbk')
        self.nb_of_links = len(link_df)
        for i in range(self.nb_of_links):
            link = Link()
            link.link_id = link_df.loc[i,'link_id']
            link.link_type = link_df.loc[i, 'link_type']
            link.from_node_id = link_df.loc[i,'from_node_id']
            link.to_node_id = link_df.loc[i,'to_node_id']
            link.built_cost = link_df.loc[i,'built_cost']
            link.trans_cost = link_df.loc[i,'trans_cost']

            link.from_node = self.node_id_to_node_dict[link.from_node_id]
            link.to_node = self.node_id_to_node_dict[link.to_node_id]
            self.link_list.append(link)

            link.from_node.m_outgoing_link_list.append(link)
            link.to_node.m_incoming_link_list.append(link)


        agent_df = pd.read_csv(os.path.join(self.cwd,'input_agent.csv'), encoding='gbk')
        self.nb_of_agents = len(agent_df)
        for i in range(self.nb_of_agents):
            agent = Agent()
            agent.agent_id = agent_df.loc[i,'agent_id']
            agent.agent_type = agent_df.loc[i, 'agent_type']
            agent.origin_node_id = agent_df.loc[i, 'origin_node_id']
            agent.destination_node_id = agent_df.loc[i, 'destination_node_id']
            agent.customized_cost_link_type = agent_df.loc[i, 'customized_cost_link_type']
            agent.flow = agent_df.loc[i, 'customized_cost_link_value']

            set_of_allowed_link_types = agent_df.loc[i, 'set_of_allowed_link_types']
            agent.set_of_allowed_links_types = list(map(int, set_of_allowed_link_types.split(";")))

            agent.origin_node = self.node_id_to_node_dict[agent.origin_node_id]
            agent.destination_node = self.node_id_to_node_dict[agent.destination_node_id]
            self.agent_list.append(agent)


        t1 = time.time()
        total_time = round(t1-t0, 2)
        print(f'  {self.nb_of_nodes} nodes, {self.nb_of_links} links, {self.nb_of_agents} agents loaded')
        print(f'  time used: {total_time}s')


    def initialization(self):
        print('Start initialization...')
        t0 = time.time()

        self.nb_of_orig_building = len(self.building_orig_list)
        self.nb_of_orig_location = len(self.location_orig_list)
        self.nb_of_dest_building = len(self.building_dest_list)
        self.nb_of_dest_location = len(self.location_dest_list)

        self.flow_mat = np.zeros([self.nb_of_orig_building, self.nb_of_dest_building])
        self.trans_cost_mat = np.zeros([self.nb_of_orig_location, self.nb_of_dest_location])
        self.build_cost_orig_mat = np.zeros([self.nb_of_orig_building, self.nb_of_orig_location])
        self.build_cost_dest_mat = np.zeros([self.nb_of_dest_location, self.nb_of_dest_building])
        self.GLB_cost_mat = np.zeros([self.nb_of_orig_building, self.nb_of_orig_location])

        node_id_unit = self.args['node_id_unit']

        for link in self.link_list:
            if (link.link_type == 1) or (link.link_type == 4):  # transportation link
                k_ind = np.mod(link.from_node_id, node_id_unit)
                l_ind = np.mod(link.to_node_id, node_id_unit)
                self.trans_cost_mat[k_ind - 1][l_ind - 1] = link.trans_cost
            elif link.link_type == 2:  # 'building_orig':
                i_ind = np.mod(link.from_node_id, node_id_unit)
                k_ind = np.mod(link.to_node_id, node_id_unit)
                self.build_cost_orig_mat[i_ind - 1][k_ind - 1] = link.built_cost
            elif link.link_type == 3:  # 'building_dest':
                l_ind = np.mod(link.from_node_id, node_id_unit)
                j_ind = np.mod(link.to_node_id, node_id_unit)
                self.build_cost_dest_mat[l_ind - 1][j_ind - 1] = link.built_cost

        for agent in self.agent_list:
            i_ind = np.mod(agent.origin_node_id, node_id_unit)
            j_ind = np.mod(agent.destination_node_id, node_id_unit)
            self.flow_mat[i_ind - 1][j_ind - 1] = agent.flow

        for link in self.link_list:
            if link.link_type == 2:  # 'building_orig'
                branch = Branch(link.from_node_id, link.to_node_id)
                i_ind = np.mod(branch.from_node_id, node_id_unit) - 1
                k_ind = np.mod(branch.to_node_id, node_id_unit) - 1
                branch.i_ind, branch.k_ind = i_ind, k_ind

                branch.trans_cost_array = self.trans_cost_mat[k_ind,:]
                branch.flow_array = self.flow_mat[i_ind,:]

                branch.branch_id = self.nb_of_branch
                self.nb_of_branch += 1

                self.branch_list.append(branch)

        t1 = time.time()
        total_time = round(t1-t0, 2)
        print(f'  time used: {total_time}s')
