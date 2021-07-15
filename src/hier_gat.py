'''
Source:
    - https://github.com/Diego999/pyGAT (Pytorch implementation of GAT)
    - https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html (Pytorch DGL GAT implementation)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import time

class HierGATLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_team_features, pos_connect):
        super(HierGATLayer, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_team_features = num_team_features
        self.pos_connect = pos_connect

        # fully connected layer - W
        self.fc = nn.Linear(in_dim, emb_dim, bias=False)
        # attention layer - between coordinators and position coaches
        self.attn_fc1 = nn.Linear(2 * emb_dim, 1, bias=False)
        # attention layer - between hc and coordinators
        self.attn_fc2 = nn.Linear(2 * emb_dim, 1, bias=False)
        # attention layer - between position coaches and position coaches
        self.attn_fc3 = nn.Linear(2 * emb_dim, 1, bias=False)

        # fully connected layer at output level
        self.output_fc2 = nn.Linear(emb_dim + num_team_features, 1, bias=True)

        self.reset_parameters()

        # dropout layer
        self.dropout = nn.Dropout(0.6)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc3.weight, gain=gain)

        nn.init.xavier_normal_(self.output_fc2.weight, gain=gain)

    def generate_outedge_edgelist(self, g, edgelist, source):
        sub_edgelist = edgelist[edgelist.source == source]
        sub_edgelist.reset_index(drop=True, inplace=True)
        source_z = sub_edgelist.source.apply(lambda x: g.nodes[x]["z"])
        target_z = sub_edgelist.target.apply(lambda x: g.nodes[x]["z"])

        sub_edgelist = sub_edgelist.assign(source_z = source_z)
        sub_edgelist = sub_edgelist.assign(target_z = target_z)

        return sub_edgelist

    def compute_attn_coef(self, edgelist, edge_type):
        source_z_stacked = torch.stack(edgelist.source_z.tolist())
        target_z_stacked = torch.stack(edgelist.target_z.tolist())
        z2 = torch.cat([source_z_stacked, target_z_stacked], dim=1)
        if edge_type == "coord_pos":
            e = F.leaky_relu(self.attn_fc1(z2))
        elif edge_type == "hc_coord":
            e = F.leaky_relu(self.attn_fc2(z2))
        elif edge_type == "pos_pos":
            e = F.leaky_relu(self.attn_fc3(z2))
        # compute alpha - softmax of e
        alpha = self.dropout(F.softmax(e, dim=0))

        return alpha

    def aggregate_neighbors(self, edgelist, edge_type):
        alpha = self.compute_attn_coef(edgelist, edge_type)
        attn_neighbor_z = torch.mul(torch.stack(edgelist.target_z.tolist()), alpha)
        aggregated_neighbors = attn_neighbor_z.sum(dim=0)

        return aggregated_neighbors

    def forward(self, g, f, team_features_dict, team_labels_dict):
        start_time = time.time()
        z = self.fc(f)

        nx.set_node_attributes(g, name="z", values=dict(zip(g.nodes(), z)))

        # split into teams
        teams = nx.weakly_connected_components(g)
        team_emb_tensor = torch.zeros((nx.number_weakly_connected_components(g), self.emb_dim))
        team_features_tensor = torch.zeros((nx.number_weakly_connected_components(g), self.num_team_features))
        team_label_tensor = torch.zeros((nx.number_weakly_connected_components(g), 1))

        # iterate through every team
        for idx, team in enumerate(teams):
            team_sub_G = g.subgraph(team).copy()
            team_edgelist = nx.to_pandas_edgelist(team_sub_G)
            
            if self.pos_connect == True:
                # position pairwise edges
                position_ids = [k for k in team_sub_G.nodes() if team_sub_G.nodes[k]["final_position"] in ["O", "D", "S"]]
                for pos in position_ids:
                    position_to_position = self.generate_outedge_edgelist(g, team_edgelist, pos)
                    aggregated_pos_z = self.aggregate_neighbors(position_to_position, "pos_pos")
                    g.nodes[pos]["z"] = F.elu(aggregated_pos_z)

            # coord -> position edges
            coord_ids = [k for k in team_sub_G.nodes() if team_sub_G.nodes[k]["final_position"] in ["OC", "SC", "DC"]]
            for coord in coord_ids:
                coord_to_position = self.generate_outedge_edgelist(g, team_edgelist, coord)
                aggregated_coord_z = self.aggregate_neighbors(coord_to_position, "coord_pos")
                g.nodes[coord]["z"] = F.elu(aggregated_coord_z)

            # hc -> coord edges
            hc = [k for k in team_sub_G.nodes() if team_sub_G.nodes[k]["final_position"] == "HC"][0]
            year = team_sub_G.nodes[hc]["Year"]
            team = team_sub_G.nodes[hc]["Team"]
            hc_to_coord = self.generate_outedge_edgelist(g, team_edgelist, hc)
            aggregated_hc_z = self.aggregate_neighbors(hc_to_coord, "hc_coord")
            g.nodes[hc]["z"] = aggregated_hc_z

            # Team embedding
            team_emb_tensor[idx,:] = aggregated_hc_z
            if self.num_team_features == 0:
                team_features_tensor = torch.Tensor()
            else:
                team_features_tensor[idx,:] = team_features_dict[(year,team)]
            team_label_tensor[idx] = team_labels_dict[(year, team)]

        return team_emb_tensor, team_features_tensor, team_label_tensor

class MultiHeadLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_heads, num_team_features, merge, pos_connect):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(HierGATLayer(in_dim, emb_dim, num_team_features, pos_connect))

        self.merge = merge

    def forward(self, g, f, team_features_dict, team_labels_dict):
        team_emb_list = []
        for attn_head in self.heads:
            team_emb, team_features, team_labels = attn_head(g, f, team_features_dict, team_labels_dict)
            team_emb_list.append(team_emb)
        if self.merge == "avg":
            return torch.mean(torch.stack(team_emb_list), dim=0), team_features, team_labels
        else:
            return torch.cat(team_emb_list, dim=1), team_features, team_labels

class HierGATTeamEmb(nn.Module):
    def __init__(self, in_dim, emb_dim, num_heads, num_team_features, merge, pos_connect):
        super(HierGATTeamEmb, self).__init__()
        self.layer1 = MultiHeadLayer(in_dim, emb_dim, num_heads, num_team_features, merge, pos_connect)
        # fully connected layer at output level
        if merge == "avg":
            self.output_layer = nn.Linear(emb_dim + num_team_features, 1, bias=True)
        else:
            self.output_layer = nn.Linear(emb_dim*num_heads + num_team_features, 1, bias=True)

    def forward(self, g, f, team_features_dict, team_labels_dict):
        team_emb, team_features, team_labels = self.layer1(g, f, team_features_dict, team_labels_dict)
        team_emb = F.elu(team_emb)
        concat_x = torch.cat((team_emb, team_features), dim=1)
        y_hat = self.output_layer(concat_x)

        return y_hat, team_labels


