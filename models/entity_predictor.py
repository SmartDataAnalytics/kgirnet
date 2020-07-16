import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityDetect(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=300, gpu=False, emb_drop=0.1, out_rel=7,
                 requires_grad=True, pretrained_emb=None, pad_idx=0):
        super(EntityDetect, self).__init__()
        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
            self.word_embed.weight.requires_grad=requires_grad

        self.n_filter = h_dim // 3
        self.h_dim = self.n_filter * 3
        self.ks = [3, 4, 5]
        self.fc_hidden = 500

        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.n_filter, (K, emb_dim)) for K in self.ks])

        self.emb_drop = nn.Dropout(emb_drop)

        # self.fc1 = nn.Linear(self.h_dim, out_rel)
        self.fc = nn.Sequential(
            nn.Dropout(emb_drop),
            nn.Linear(self.h_dim, self.fc_hidden),
            nn.PReLU(),
            nn.BatchNorm1d(self.fc_hidden),
            nn.Dropout(emb_drop),
            nn.Linear(self.fc_hidden, out_rel),
        )

        if gpu:
            self.cuda()

    def forward(self, x):
        """
        :param x: size = B X S X E
        :return:
        """
        x = self.emb_drop(self.word_embed(x))
        # x = self.emb_drop(x)
        x = x.unsqueeze(1)  # S x 1 x B x E
        # print (x.size())
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        out = F.log_softmax(self.fc(x), dim=-1)
        return out

