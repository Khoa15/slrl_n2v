import torch
from torch import nn

from .layers import Swish, make_linear_block


class Agent(nn.Module):

    def __init__(self, hidden_size, input_dim=1, norm_type=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim

        self.seed_embedding = nn.Linear(input_dim, hidden_size, bias=False)
        self.node_embedding = nn.Linear(input_dim, hidden_size, bias=False)
        self.input_mapping = nn.Sequential(
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
            make_linear_block(hidden_size, hidden_size, Swish, norm_type)
        )
        self.node_score_layer = nn.Linear(hidden_size, 1, bias=False)
        self.stopping_score_layer = nn.Linear(hidden_size, 2, bias=False)

        nn.init.zeros_(self.node_score_layer.weight.data)
        nn.init.zeros_(self.stopping_score_layer.weight.data)

    def forward(self, x_seeds, x_nodes, indptr):
        # Ensure dimensions match [Batch, Dim]
        if x_seeds.dim() == 1:
            x_seeds = x_seeds.unsqueeze(1)
        if x_nodes.dim() == 1:
            x_nodes = x_nodes.unsqueeze(1)

        h = self.seed_embedding(x_seeds) + self.node_embedding(x_nodes)
        h = self.input_mapping(h)
        node_scores = self.node_score_layer(h).squeeze(1)

        batch = []
        for startpoint, endpoint, candidate_endpoint in indptr:
            if startpoint == endpoint:
                raise ValueError('Finished Episode!')
            else:
                stop_node = torch.mean(h[startpoint:endpoint], dim=0).unsqueeze(0)
                # 社区的value
                node_logits = torch.log_softmax(node_scores[startpoint:candidate_endpoint], 0)  # [r-l]
                stopping_logits = torch.log_softmax(self.stopping_score_layer(stop_node), 1).squeeze(0)  # [2]
                logits = torch.cat([node_logits + stopping_logits[0], stopping_logits[1:]], dim=0)  # [*+1]
                batch.append(logits)           # 存储当前样本各个节点log_softmax，社区的value 添加到 batch 列表中。
        batch_logits = batch
        # print(batch_logits)
        # for logits in batch_logits:
        #     print(f"type: {type(logits)}")
        # print(batch)
        # batch_logits = torch.stack(batch)
        return batch_logits