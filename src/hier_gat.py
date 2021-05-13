'''
Source:
    - https://github.com/Diego999/pyGAT (Pytorch implementation of GAT)
    - https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html (Pytorch DGL GAT implementation)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

class HierGATLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, dropout):
        super(HierGATLayer, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout = dropout

        # fully connected layer - W
        self.fc = nn.Linear(in_dim, emb_dim, bias=False)

        self.attn_fc = nn.Linear(2 * emb_dim, 1, bias=True)
        # attention layer - between coordinators and position coaches
        # self.attn_fc1 = nn.Linear(2 * emb_dim, 1, bias=False)
        # attention layer - between hc and coordinators
        # self.attn_fc2 = nn.Linear(2 * emb_dim, 1, bias=False)

        # fully connected layer at output level
        # self.output_fc1 = nn.Linear(emb_dim+1, int(emb_dim/2), bias=True)
        # self.output_fc2 = nn.Linear(int(emb_dim/2), 2, bias=True)
        self.output_fc2 = nn.Linear(emb_dim+1, 2, bias=True)

        self.reset_parameters()

        # dropout layer
        self.dropout = nn.Dropout(0.3)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc1.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc2.weight, gain=gain)
        # nn.init.xavier_normal_(self.output_fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.output_fc2.weight, gain=gain)

    def generate_coord_position_edgelist(self, g):
        # define the nodes with coordinator positions
        coord_ids = [k for k in g.nodes() if g.nodes[k]["final_position"] in ["OC", "SC", "DC"]]

        # extract edgelist of coordinators -> position coaches
        edgelist = nx.to_pandas_edgelist(g)
        coord_to_position = edgelist[edgelist.source.isin(coord_ids)]
        coord_to_position.reset_index(drop=True, inplace=True)

        # assign z of source and target
        source_z = coord_to_position.source.apply(lambda x: g.nodes[x]["z"])
        target_z = coord_to_position.target.apply(lambda x: g.nodes[x]["z"])

        coord_to_position = coord_to_position.assign(source_z = source_z)
        coord_to_position = coord_to_position.assign(target_z = target_z)
        
        return coord_to_position

    def generate_hc_coord_edgelist(self, g):
        hc_ids = [k for k in g.nodes() if g.nodes[k]["final_position"] == "HC"]
        # extract edgelist of head coach -> coordinator
        edgelist = nx.to_pandas_edgelist(g)
        hc_to_coord = edgelist[edgelist.source.isin(hc_ids)]
        hc_to_coord.reset_index(drop=True, inplace=True)

        # assign z of source and target
        source_z = hc_to_coord.source.apply(lambda x: g.nodes[x]["z"])
        target_z = hc_to_coord.target.apply(lambda x: g.nodes[x]["z"])

        hc_to_coord = hc_to_coord.assign(source_z = source_z)
        hc_to_coord = hc_to_coord.assign(target_z = target_z)

        return hc_to_coord

    def compute_e_func(self, row):
        source_z = row.source_z
        target_z = row.target_z

        z2 = torch.cat([source_z, target_z], dim=0)
        e = self.dropout(F.leaky_relu(self.attn_fc(z2)))
        # if edge_type == "coord_position":
            # e = self.dropout(F.leaky_relu(self.attn_fc1(z2)))
        # elif edge_type == "hc_coord":
            # e = self.dropout(F.leaky_relu(self.attn_fc2(z2)))

        return e[0]

    def compute_attn_coef(self, g, sub_edgelist):
        
        # get the list of coord->position edges
        sub_edges = list(zip(sub_edgelist.source, sub_edgelist.target))
        # tensor to store the attention coefficient for each coord->position edge

        sub_edgelist_e = sub_edgelist.apply(self.compute_e_func, axis=1)
        sub_edgelist = sub_edgelist.assign(e=sub_edgelist_e)
        e_tensor = torch.Tensor(sub_edgelist_e).view(sub_edgelist_e.shape[0], 1)

        # assign e to edge attributes
        nx.set_edge_attributes(g, name="e", values=dict(zip(sub_edges, e_tensor)))
        
        # softmax of the attention coefficient
        # group the coord->position edges by sources
        groups = sub_edgelist.groupby('source')
        alpha = torch.zeros((e_tensor.shape[0], 1))
        for source, group in groups:
            edge_attn_coef = e_tensor[group.index,:]
            softmax_edge_attn_coef = F.softmax(edge_attn_coef, dim=0)
            alpha[group.index,:] = softmax_edge_attn_coef
            
        # assign the alpha values for sub edges
        nx.set_edge_attributes(g, name="alpha", values=dict(zip(sub_edges, alpha)))
        return g, sub_edgelist

    def aggregate_neighbors(self, g, edgelist):
        sources = edgelist.source.unique()
        for source in sources:
            group = edgelist[edgelist.source == source]
            neighbor_tensors = []
            for neighbor in group.target:
                alpha = g.edges[source, neighbor]["alpha"]
                neighbor_z = g.nodes[neighbor]["z"]
                attn_z = alpha * neighbor_z
                neighbor_tensors.append(attn_z)
            sum_neighbor_tensors = torch.stack(neighbor_tensors, dim=0).sum(dim=0)
            sum_neighbor_tensors = F.relu(sum_neighbor_tensors)
            g.nodes[source]["z"] = sum_neighbor_tensors

        return g
    
    def team_embedding(self, g):
        hc_ids = [k for k in g.nodes() if g.nodes[k]["final_position"] == "HC"]
        team_emb = torch.zeros((len(hc_ids), self.emb_dim))
        for idx, hc in enumerate(hc_ids):
            hc_emb = g.nodes[hc]["z"]
            team_emb[idx,:] = hc_emb

        return g, team_emb

    def forward(self, g, f, salary):
        if self.dropout == True:
            z = self.dropout(F.relu(self.fc(f)))
        else:
            z = F.relu(self.fc(f))

        nx.set_node_attributes(g, name="z", values=dict(zip(g.nodes(), z)))

        coord_to_position_edgelist = self.generate_coord_position_edgelist(g)
        g, coord_to_position = self.compute_attn_coef(g, coord_to_position_edgelist)
        g = self.aggregate_neighbors(g, coord_to_position)

        hc_to_coord_edgelist = self.generate_hc_coord_edgelist(g)
        g, hc_to_coord = self.compute_attn_coef(g, hc_to_coord_edgelist)
        g = self.aggregate_neighbors(g, hc_to_coord)

        g, team_emb = self.team_embedding(g)

        team_emb_salary = torch.cat((team_emb, salary), dim=1)
        # y_hat = self.dropout(F.relu(self.output_fc1(team_emb_salary)))
        # y_hat = self.output_fc2(y_hat)
        y_hat = F.relu(self.output_fc2(team_emb_salary))
        y_hat = F.softmax(y_hat, dim=1)

        return g, y_hat




