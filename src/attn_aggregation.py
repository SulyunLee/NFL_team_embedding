
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

class AttnAggLayer(nn.Module):
    def __init__(self, in_dim, emb_dim, num_team_features):
        super(AttnAggLayer, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_team_features = num_team_features

        self.attn_fc = nn.Linear(in_dim*2, 1, bias=False)
        self.fc = nn.Linear(in_dim, emb_dim, bias=False)
        self.output_fc = nn.Linear(emb_dim+num_team_features, 1, bias=True)

        self.dropout = nn.Dropout(0.6)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.output_fc.weight, gain=gain)

    def forward(self, g, f, team_features_dict, team_labels_dict, number_of_teams):
        # split into teams
        teams = nx.strongly_connected_components(g)

        team_emb_tensor = torch.zeros((number_of_teams, self.emb_dim))
        team_features_tensor = torch.zeros((number_of_teams, self.num_team_features))
        team_label_tensor = torch.zeros((number_of_teams, 1))

        # iterate through every team
        for idx, team in enumerate(teams):
            team_sub_G = g.subgraph(team).copy()

            # get the edgelist of team
            team_edgelist = nx.to_pandas_edgelist(team_sub_G)

            team_zs = []

            # retrieve the features
            source_f = team_edgelist.source.apply(lambda x:g.nodes[x]["f"])
            target_f = team_edgelist.target.apply(lambda x: g.nodes[x]["f"])

            team_edgelist = team_edgelist.assign(source_f=source_f)
            team_edgelist = team_edgelist.assign(target_f=target_f)

            # compute attention coefficients
            source_f_stacked = torch.stack(team_edgelist.source_f.tolist())
            target_f_stacked = torch.stack(team_edgelist.target_f.tolist())
            f2 = torch.cat([source_f_stacked, target_f_stacked], dim=1)
            e = F.leaky_relu(self.attn_fc(f2))
            team_edgelist = team_edgelist.assign(e=e.detach().numpy())

            # get the edgelist of each source node
            for node in team_sub_G.nodes():
                if team_sub_G.nodes[node]["final_position"] == "HC":
                    year = team_sub_G.nodes[node]["Year"]
                    team = team_sub_G.nodes[node]["Team"]
                    
                sub_edgelist = team_edgelist[team_edgelist.source == node]
                sub_edgelist.reset_index(drop=True, inplace=True)

                alpha = self.dropout(F.softmax(torch.Tensor(sub_edgelist.e), dim=0))
                alpha = alpha.view(alpha.shape[0],1)

                # aggregate neighborhood features with attention
                attn_neighbor_f = torch.mul(torch.stack(sub_edgelist.target_f.tolist()), alpha)
                aggregated_neighbors = attn_neighbor_f.sum(dim=0)
                g.nodes[node]["z"] = aggregated_neighbors
                team_zs.append(aggregated_neighbors)

            # team embedding
            z_prime = F.elu(self.fc(torch.stack(team_zs)))
            team_emb = z_prime.mean(dim=0)
            team_emb_tensor[idx,:] = team_emb
            if self.num_team_features == 0:
                team_features_tensor = torch.Tensor()
            else:
                team_features_tensor[idx,:] = team_features_dict[(year,team)]
            team_label_tensor[idx] = team_labels_dict[(year,team)]

        concat_x = torch.cat((team_emb_tensor, team_features_tensor), dim=1)
        y_hat = self.output_fc(concat_x)

        return g, y_hat, team_label_tensor







