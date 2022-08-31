import torch
import torch.nn as nn

class my_bert_model(nn.Module):
    def __init__(self, args, bert_model):
        super(my_bert_model, self).__init__()
        self.args = args
        self.bert = bert_model

    # @staticmethod
    def forward(self, **inputs):
        outputs = self.bert(**inputs)
        hidden_states = outputs[0]
        last_four_hidden_states = torch.flatten(hidden_states[:,-4:,:],1)
        return last_four_hidden_states

class my_gru(nn.Module):
    def __init__(self, args):
        super(my_gru, self).__init__()

        # network configure
        self.args = args

        # pos embedding
        self.GRU_layers= args.GRU_layers
        self.GRU_directions = 2 if args.bidirectional else 1
        self.embed_size=3072
        self.nhid_pos=512


        self.label_representation_maker = nn.GRU(input_size=self.embed_size, hidden_size=self.nhid_pos,
                              num_layers=self.GRU_layers, batch_first=True, 
                              dropout = self.args.dropout,
                              bidirectional=args.bidirectional)
        
        self.postGRU = nn.GRU(input_size=512, hidden_size=self.nhid_pos,
                              num_layers=1, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(self.nhid_pos * self.GRU_directions * self.GRU_layers,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,5)) ###if for THUCNews dataset:nn.Linear(128,14)

   



    # @staticmethod
    def forward(self, inputs):
        output_GRU1, hidden_l1 = self.label_representation_maker(inputs)
        output_GRU, hidden_l= self.postGRU(output_GRU1)
        # #展平
        # print(output_GRU.shape)
        # print(hidden_l.shape)
        hidden_l = hidden_l.permute(1,0,2)
        hidden_l = hidden_l.reshape(hidden_l.shape[0],self.nhid_pos * self.GRU_directions * self.GRU_layers)

        final_out = self.linear(hidden_l)

        return final_out