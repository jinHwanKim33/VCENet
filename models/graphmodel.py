from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput


class GraphModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(GraphModel, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)

        ## choi - detector ##
        self.detection_alg = args.detection_alg
        if self.detection_alg == 'fasterrcnn':
            graph_input = 518
        elif self.detection_alg == 'detr':
            graph_input = 262
        else:
            raise NotImplementedError()

        self.graph_detection_feature = nn.Sequential(
            nn.Linear(graph_input, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )
        ## end ##

        # self.graph_detection_feature = nn.Sequential(
        #     nn.Linear(518, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 49),
        # )

        self.graph_detection_other_info_linear_1 = nn.Linear(6, self.num_cate)
        self.graph_detection_other_info_linear_2 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_3 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_4 = nn.Linear(self.num_cate, self.num_cate)
        self.graph_detection_other_info_linear_5 = nn.Linear(self.num_cate, self.num_cate)

        self.embed_action = nn.Linear(action_space, 10)

        # pointwise_in_channels = 64 + self.num_cate + 10

        ## choi - w/o global ##
        self.state_scope = args.state_scope
        if self.state_scope == 'all':
            pointwise_in_channels = 64 + self.num_cate + 10  # for all : 96
        elif self.state_scope == 'local':
            pointwise_in_channels = self.num_cate + 10  # for w/o global : 32
        elif self.state_scope == 'global':
            pointwise_in_channels = 64 + 10  # for w/o local : 74
        else:
            raise NotImplementedError()
        ## end ##

        print("pointwise_in_channels", pointwise_in_channels)
        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0     #1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_embedding_input):

        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

        ## choi - w/o global ##
        image_embedding = F.relu(self.conv1(state))
        if self.state_scope == 'all':
            target_info = torch.cat((target['info'], target['indicator']), dim=1)
            target_info = F.relu(self.graph_detection_other_info_linear_1(target_info))
            target_info = target_info.t()
            target_info = F.relu(self.graph_detection_other_info_linear_2(target_info))
            target_info = F.relu(self.graph_detection_other_info_linear_3(target_info))
            target_info = F.relu(self.graph_detection_other_info_linear_4(target_info))
            target_info = F.relu(self.graph_detection_other_info_linear_5(target_info))
            target_appear = torch.mm(target['appear'].t(), target_info).t()
            target = torch.cat((target_appear, target['info'], target['indicator']), dim=1)

            target = F.relu(self.graph_detection_feature(target))
            target_embedding = target.reshape(1, self.num_cate, 7, 7)

            x = self.dropout(image_embedding)
            x = torch.cat((x, target_embedding, action_reshaped), dim=1)  # 64 + 22(cate) + 10
        elif self.state_scope == 'local':
            target_info = torch.cat((target['info'], target['indicator']), dim=1)
            target_info = F.relu(self.graph_detection_other_info_linear_1(target_info))
            target_info = target_info.t()
            target_info = F.relu(self.graph_detection_other_info_linear_2(target_info))
            target_info = F.relu(self.graph_detection_other_info_linear_3(target_info))
            target_info = F.relu(self.graph_detection_other_info_linear_4(target_info))
            target_info = F.relu(self.graph_detection_other_info_linear_5(target_info))
            target_appear = torch.mm(target['appear'].t(), target_info).t()
            target = torch.cat((target_appear, target['info'], target['indicator']), dim=1)

            target = F.relu(self.graph_detection_feature(target))
            target_embedding = target.reshape(1, self.num_cate, 7, 7)

            x = torch.cat((target_embedding, action_reshaped), dim=1)  # 22(cate) + 10
        elif self.state_scope == 'global':
            x = self.dropout(image_embedding)
            x = torch.cat((x, action_reshaped), dim=1)  # 64 + 10
        else:
            raise NotImplementedError()
        ## end ##
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs

        x, image_embedding = self.embedding(state, target, action_probs)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
