# imports
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import f1_score
from torch import optim
from torch.autograd import Variable
# from detector_utils import masked_cross_entropy
from utils.io_utils import masked_cross_entropy, top_k_top_p_filtering


class Seq2Seq(nn.Module):
    """
    Sequence to sequence model with Attention
    """
    def __init__(self, hidden_size, max_r, n_words, decoder_lr_ration, b_size, sos_tok, eos_tok, itos, emb_dim,
                 inp_graph_feat_s, kb_max_size, gpu=False, lr=0.001, n_layers=1, clip=4.0, pretrained_emb=None,
                 dropout=0.1, rnn_dropout=0.3, emb_drop=0.5, teacher_forcing_ratio=5.0, first_kg_token=4740,
                 topp=20, topk=0.9, n_g_inp_feat=1, use_graph=False):
        super(Seq2Seq, self).__init__()
        self.name = "GraphLaplacian"
        self.output_size = n_words
        self.hidden_size = hidden_size
        self.inp_graph_feat_size = inp_graph_feat_s
        self.topp = topp
        self.topk = topk
        self.n_inp_g_feat = n_g_inp_feat
        self.max_r = max_r  # max response len
        self.lr = lr
        self.emb_dim = emb_dim
        self.first_kg_token = first_kg_token
        self.decoder_learning_ratio = decoder_lr_ration
        self.kb_max_size = kb_max_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_drop = rnn_dropout
        self.emb_drop = emb_drop
        self.b_size = b_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_tok = sos_tok
        self.eos_tok = eos_tok
        self.itos = itos
        self.clip = clip
        self.use_cuda = gpu
        self.sentient_loss = nn.BCEWithLogitsLoss()
        self.kg_loss = nn.BCEWithLogitsLoss()
        self.kg_kld = nn.KLDivLoss()

        # Indexes for output vocabulary
        if self.use_cuda:
            self.kg_vocab = torch.from_numpy(np.arange(first_kg_token, first_kg_token+self.kb_max_size)).long().cuda()
        else:
            self.kg_vocab = torch.from_numpy(np.arange(first_kg_token, first_kg_token+self.kb_max_size)).long()

        # Use single RNN for both encoder and decoder
        # self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        # initializing the model
        self.encoder = EncoderRNN(n_layers=self.n_layers, emb_size=self.emb_dim, dropout=self.rnn_drop, hidden_size=self.hidden_size, vocab_size=self.output_size,
                                 gpu=self.use_cuda, pretrained_emb=pretrained_emb)
        self.decoder = Decoder(hidden_size=self.hidden_size, emb_dim=self.emb_dim, vocab_size=self.output_size, dropout=self.rnn_drop)
        self.dropout = nn.Dropout(dropout)
        self.graph_g = GraphLaplacian(kb_out_max=self.kb_max_size, dropout=dropout, inp_dim=self.inp_graph_feat_size, hidden_dec=self.hidden_size)

        self.kg_trainer = nn.Linear(self.inp_graph_feat_size*self.n_inp_g_feat, self.kb_max_size)
        self.maxpool = nn.MaxPool1d(int(self.kb_max_size*self.n_inp_g_feat))
        nn.init.xavier_normal_(self.kg_trainer.weight)
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            # self.embedding = self.embedding.cuda()
            self.graph_g = self.graph_g.cuda()
            self.kg_trainer = self.kg_trainer.cuda()
            self.maxpool = self.maxpool.cuda()
            # self.rnn = self.rnn.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)

        self.loss = 0
        self.print_every = 1

    # def train_batch(self, input_batch, input_chunk, out_batch, input_mask, target_mask, kb, kb_mask, sentient_orig):
    def train_batch(self, input_batch, input_mask, input_graph, s_gate, out_kg, out_kg_mask, response_out, response_mask):

        self.encoder.train(True)
        self.decoder.train(True)
        self.graph_g.train(True)
        self.kg_trainer.train(True)
        b_size = input_batch.size(0)
        # print (b_size)
        # print (self.kb_max_size)
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # loss_Vocab,loss_Ptr,loss_Gate = 0,0,0
        # Run words through encoder
        # input_len = torch.sum(input_mask, dim=0)
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_mask)
        # maxpool_inp = self.maxpool(torch.sigmoid(input_graph.unsqueeze(0))).squeeze(0)
        # input_graph = input_graph.view(b_size, (self.kb_max_size*self.inp_graph_feat_size))
                                                                                #  Reshape inp graph with more features
        # kg_out = torch.sigmoid(self.dropout(self.kg_trainer(input_graph))) * out_kg_mask
        # kg_out = torch.sigmoid(self.dropout(self.kg_trainer(input_graph))) * out_kg_mask
        # target_len = torch.sum(target_mask, dim=0)
        response_out = response_out.transpose(0, 1)  # B X S --> S X B
        s_gate = s_gate.transpose(0, 1)  # B X S --> S X B
        target_len = response_out.size(0)  # Get S
        # print (min(max(target_len), self.max_r))
        max_target_length = min(target_len, self.max_r)
        # print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())

        # Prepare input and output variables
        if self.use_cuda:
            decoder_input = torch.tensor([self.sos_tok] * b_size).cuda().long()
            sentinel_values = torch.zeros(int(max_target_length), b_size).cuda()
            kg_out_seq = torch.zeros(int(max_target_length), b_size, self.kb_max_size).cuda()
            all_decoder_outputs_vocab = torch.zeros(int(max_target_length), b_size, self.output_size).cuda()
            only_vocab_mask = torch.ones(b_size, self.output_size).cuda()  # mask to remove all entities not in input graph
        else:
            decoder_input = torch.tensor([self.sos_tok] * b_size).long()
            sentinel_values = torch.zeros(int(max_target_length), b_size)
            kg_out_seq = torch.zeros(int(max_target_length), b_size, self.kb_max_size)
            all_decoder_outputs_vocab = torch.zeros(int(max_target_length), b_size, self.output_size)
            only_vocab_mask = torch.ones(b_size, self.output_size) # mask to remove all entities not in input graph

        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        only_vocab_mask[:, -self.kb_max_size:] = 0.0
        # all_vocab_mask[all_vocab_mask==0] = -100  # add - value for objects which are not present to add penalty
        # Choose whether to use teacher forcing
        use_teacher_forcing = random.randint(0, 10) < self.teacher_forcing_ratio

        if use_teacher_forcing:
            for t in range(max_target_length):

                decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)  # B X H & TUPLE
                # print (decoder_vocab.size(), decoder_hidden.size())
                # sentient gatingkg_vocab
                d_h, d_c = decoder_hidden
                # sentient_gate = self.graph_g(d_h.squeeze(0), s_gate[t - 1], maxpool_inp)
                # s = sentient_orig[t].reshape(b_size, 1)

                # s = torch.sigmoid(sentient_gate)
                # obj = s * input_graph
                # kg_out_seq[t] = obj
                # print (obj[1][1])
                # obj_np = obj.detach().numpy()
                # decoder_vocab = decoder_vocab * only_vocab_mask
                # decoder_vocab = (1 - s) * decoder_vocab
                # print(d44444ecoder_vocab[1][1730])
                # decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, self.kb_max_size), obj)
                # decoder_vocab_np = decoder_vocab.numpy()
                # print(decoder_vocab[1][1730])
                # decoder_vocab = [top_k_top_p_filtering(vocab_dist, top_k=self.topp, top_p=self.topk) for vocab_dist in
                #                 decoder_vocab]
                # decoder_vocab = torch.stack(decoder_vocab)
                # sentinel_values[t] = s.squeeze()
                all_decoder_outputs_vocab[t] = decoder_vocab
                decoder_input = response_out[t].long()  # Next input is current target
        else:
            # print ('Not TF..')
            for t in range(max_target_length):
                # inp_emb_d = self.embedding(decoder_input)
                decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
                d_h, d_c = decoder_hidden
                sentient_gate = self.graph_g(d_h.squeeze(0), s_gate[t - 1], maxpool_inp)
                s = torch.sigmoid(sentient_gate)
                obj = s * input_graph
                kg_out_seq[t] = obj
                decoder_vocab = decoder_vocab * only_vocab_mask
                decoder_vocab = (1 - s) * decoder_vocab
                decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, self.kb_max_size), obj)
                # decoder_vocab = [top_k_top_p_filtering(vocab_dist, top_k=self.topp, top_p=self.topk) for vocab_dist in decoder_vocab]
                # decoder_vocab = torch.stack(decoder_vocab)
                # print (decoder_vocab.size())
                all_decoder_outputs_vocab[t] = decoder_vocab
                sentinel_values[t] = s.squeeze()
                topv, topi = decoder_vocab.data.topk(1)  # get prediction from decoder
                decoder_input = topi.view(-1)  # use this in the next time-steps

        # print (all_decoder_outputs_vocab.size(), out_batch.size())
        # out_batch = out_batch.transpose(0, 1).contiguous
        # target_mask = target_mask.transpose(0, 1).contiguous()
        # print (all_decoder_outputs_vocab.size(), out_batch.size(), target_mask.size())
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
            response_out.transpose(0, 1).contiguous(),  # -> B x S
            response_mask
        )
        kg_out_seq = torch.sum(kg_out_seq, dim=0)
        # sentient_loss = self.sentient_loss(sentinel_values, s_gate)
        # kg_loss = self.kg_loss(kg_out_seq, out_kg)
        # kg_loss_kl = self.kg_kld(kg_out_seq, out_kg)

        # loss = 0.5* loss_Vocab + 0.2*sentient_loss + 0.3*kg_loss
        loss = loss_Vocab
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        #  Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()

    def evaluate_batch(self, input_batch, input_mask, input_graph, s_gate, out_kg_mask, response_out, response_mask):
        """
        evaluating batch
        :param input_batch:
        :param out_batch:
        :param input_mask:
        :param target_mask:
        :return:
        """
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        self.graph_g.train(True)
        self.kg_trainer.train(True)
        # self.embedding.train(False)

        # inp_emb = self.embedding(input_batch)
        # output decoder words
        b_size = input_batch.size(0)
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_mask)
        maxpool_inp = self.maxpool(torch.sigmoid(input_graph.unsqueeze(0))).squeeze(0)
        # input_graph = input_graph.view(b_size, self.kb_max_size * self.inp_graph_feat_size)  # Reshape inp graph with more features
        # kg_out = torch.sigmoid(self.kg_trainer(input_graph) * maxpool_inp) * out_kg_mask
        kg_out = torch.sigmoid(self.dropout(self.kg_trainer(input_graph))) * out_kg_mask
        # kg_out = torch.relu(self.dropout(torch.sigmoid(input_graph))) * out_kg_mask
        # kg_out = (self.dropout(torch.sigmoid(input_graph))) * out_kg_mask
        b_size = input_batch.size(0)
        # target_len = torch.sum(target_mask, dim=0)
        target_len = response_out.size(1)
        response_out = response_out.transpose(0, 1)  # B X S --> S X B
        s_gate = s_gate.transpose(0, 1)  # B X S --> S X B
        # print (min(max(target_len), self.max_r))
        max_target_length = (min(target_len, self.max_r))
        # print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())

        # Prepare input and output variables

        if self.use_cuda:
            decoder_input = torch.tensor([self.sos_tok] * b_size).cuda().long()
            sentinel_values = torch.zeros(int(max_target_length), b_size).cuda()
            all_decoder_outputs_vocab = torch.zeros(int(max_target_length), b_size, self.output_size).cuda()
            only_vocab_mask = torch.ones(b_size,
                                        self.output_size).cuda()  # mask to remove all entities not in input graph
        else:
            decoder_input = torch.tensor([self.sos_tok] * b_size).long()
            sentinel_values = torch.zeros(int(max_target_length), b_size)
            all_decoder_outputs_vocab = torch.zeros(int(max_target_length), b_size, self.output_size)
            only_vocab_mask = torch.ones(b_size, self.output_size) # mask to remove all entities not in input graph

        decoded_words = torch.zeros(int(max_target_length), b_size).cuda() if self.use_cuda else \
                        torch.zeros(int(max_target_length), b_size)
        only_vocab_mask[:, -self.kb_max_size:] = out_kg_mask
        # all_vocab_mask[all_vocab_mask == 0] = -1
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        # provide data to decoder
        for t in range(max_target_length):
            #print (decoder_input)
            #inp_emb_d = self.embedding(decoder_input)

            decoder_vocab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
            all_decoder_outputs_vocab[t] = decoder_vocab
            # sentient_gate = self.graph_g(input_graph, s_gate[t - 1], maxpool_inp)
            d_h, d_c = decoder_hidden
            # sentient_gate = self.graph_g(d_h.squeeze(0), s_gate[t - 1], maxpool_inp)
            # s = torch.sigmoid(sentient_gate)
            # obj = (s * input_graph)
            # decoder_vocab = decoder_vocab * only_vocab_mask
            # decoder_vocab = (1 - s) * decoder_vocab
            # decoder_vocab = decoder_vocab.scatter_add(1, self.kg_vocab.repeat(b_size).view(b_size, self.kb_max_size),
            #                                          obj)
            # sentinel_values[t] = s.squeeze()
            # decoder_vocab = [top_k_top_p_filtering(vocab_dist, top_k=self.topp, top_p=self.topk) for vocab_dist in
            #                  decoder_vocab]
            # decoder_vocab = torch.stack(decoder_vocab)
            all_decoder_outputs_vocab[t] = decoder_vocab

            topv, topi = decoder_vocab.data.topk(1)  # get prediction from decoder
            # topi = decoder_vocab, out_kg_mask, topi)
            decoder_input = topi.view(-1)  # use this in the next time-steps
            decoded_words[t] = (topi.view(-1))

            # target_mask = target_mask.transpose(0,1).contiguous()

        loss_vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
            response_out.transpose(0, 1).contiguous(),  # -> B x S
            response_mask
        )

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        # self.embedding.train(True)

        return decoded_words, loss_vocab

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)


class EncoderRNN(nn.Module):
    """
    Encoder RNN module
    """
    def __init__(self, emb_size, hidden_size, vocab_size, n_layers=1, dropout=0.1, emb_drop=0.2,
                 gpu=False, pretrained_emb=None, train_emb=True):
        super(EncoderRNN, self).__init__()
        self.gpu = gpu
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_cuda = gpu
        # self.b_size = b_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(emb_drop)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, dropout=self.dropout, batch_first=True)
        if pretrained_emb is not None:
           self.embedding.weight.data.copy_(pretrained_emb)
        if train_emb == False:
           self.embedding.weight.requires_grad = False
        #self.rnn = rnn

    def init_weights(self, b_size):
        #intiialize hidden weights
        c0 = torch.rand(self.n_layers, b_size, self.hidden_size)
        h0 = torch.rand(self.n_layers, b_size, self.hidden_size)

        if self.gpu:
            c0 = c0.cuda()
            h0 = h0.cuda()

        return h0, c0

    def forward(self, inp_q, input_mask, input_lengths=None):
        '''
        :param inp_q: B X S X E
        :param input_mask: B X S
        :param input_lengths:
        :return: encode input query
        '''
        # input_q =numpy S X B input_mask = S X B
        # embeddeinputs, inp_mask, encoder_hidden, decoder_out, kb, kb_mask, last_sentientd = self.embedding(input_q)

        embedded = self.embedding_dropout(self.embedding(inp_q)) # B X S X E
        b_size = inp_q.size(0)
        hidden = self.init_weights(b_size)
        embedded = self.embedding_dropout(embedded)

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)

        outputs, hidden = self.rnn(embedded, hidden)  # outputs = B X S X n_layers*H, hidden = 2 * [B X 1 X H]
        outputs = outputs * input_mask.unsqueeze(-1)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        return outputs, hidden


class Attention(nn.Module):
    """
    Attention mechanism (Luong)
    """
    def __init__(self, hidden_size, hidden_size1):
        super(Attention, self).__init__()
        # weights
        self.W_h = nn.Linear(hidden_size + hidden_size1, hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.epsilon = 1e-10

        nn.init.xavier_normal_(self.W_h.weight)

    def forward(self, encoder_outputs, decoder_hidden, inp_mask):
        seq_len = encoder_outputs.size(1) # get sequence lengths S
        H = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1) # B X S X H
        energy = torch.tanh(self.W_h(torch.cat([H, encoder_outputs], 2))) # B X S X H
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B X 1 X H]
        energy = torch.bmm(v,energy).view(-1, seq_len) # [B X T]
        a = torch.softmax(energy, dim=-1) * inp_mask # B X T
        normalization_factor = a.sum(1, keepdim=True)
        a = a / (normalization_factor+self.epsilon) # adding a small offset to avoid nan values

        a = a.unsqueeze(1)
        context = a.bmm(encoder_outputs)

        return a, context


class Decoder(nn.Module):
    """
    Decoder RNN
    """
    def __init__(self, hidden_size, emb_dim, vocab_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        #self.rnn = rnn
        self.out = nn.Linear(self.hidden_size, vocab_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        # Attention
        self.attention = Attention(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.concat.weight)

    def forward(self, input_q, last_hidden, encoder_outputs, inp_mask):
        '''

        :param input_q:
        :param last_hidden:
        :param encoder_outputs: B X S X H
        :param inp_mask:
        :return:
        '''
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        # inp_emb = self.embedding(input_q)
        # print (inp_emb.size())
        batch_size = min(last_hidden[0].size(1), input_q.size(0))
        # inp_emb = inp_emb[-batch_size:]

        # max_len = encoder_outputs.size(0)args
        # encoder_outputs = encoder_outputs # B X S X H
        embedded = self.embedding(input_q)
        embedded = self.dropout(embedded)
        # print (embedded.size())

        embedded = embedded.view(1, batch_size, self.emb_dim)  # S=1 x B x N
        #print (embedded.size())
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn(embedded, last_hidden)

        s_t = hidden[0][-1].unsqueeze(0)
        # print (s_t.size())
        # s_t = (rnn_output * input_q_mask.unsqueeze(-1))[-1].squeeze()

        alpha, context = self.attention(encoder_outputs, s_t, inp_mask)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x H -> B x H
        context = context.squeeze(1)       # B x S=1 x H -> B x H
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # print (concat_output.size())
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden


class GraphLaplacian(nn.Module):
    """
    Sentinel Module
    """
    def __init__(self, kb_out_max, inp_dim, dropout, hidden_dec, cuda=False):
        super(GraphLaplacian, self).__init__()
        self.kb_size = kb_out_max
        self.sentinel_gate = nn.Linear(hidden_dec+1+1, 1)  # 1 for last prediction
        # self.out_kb = nn.Linear(self.kb_size, self.kb_size)
        nn.init.xavier_normal_(self.sentinel_gate.weight)
        # self.cosine_sim = nn.CosineSimilarity(dim=2)
        self.drop = nn.Dropout(dropout)
        # self.vocab_size = vocab_size
        # self.s_embedding = nn.Embedding(self.vocab_size, emb_dim, padding_idx=0)
        self._cuda = cuda
        # Attention

    def forward(self, hidden_dec, last_sentient, maxpooled_graph):
        # inp_graph = B X E, encoder_hidden = S X 1 X B, decoder_out = S X 1 X B, B X 1
        #                        kb_avg_emb = kb_size X B X E, kb_mask = kb_size X B
        # attention_weights = attention_weights.unsqueeze(1)  # B X S == > B X 1 X S
        # Get average embeddings for the input
        # b_s = attention_weights.size(0)
        # print (inp.size())
        # Get cosine similarity with kb subject and relations and multiply with mask
        # kb_cosine = self.cosine_sim(inp_emb_avg, kb_avg_emb)  # B X kb_size
        # kb_trainer = self.kg_out(input_graph) * out_kg_mask  # B X kb_size
        # kb_cosine = self.out_kb(F.sigmoid(kb_cosine)) * kb_mask
        # print (decoder_h.size())
        inp = torch.cat([hidden_dec, last_sentient.unsqueeze(1), maxpooled_graph], dim=-1)
        sentient = torch.relu(self.drop(self.sentinel_gate(inp)))  # B X 1
        return sentient
