# imports
import math
import random
import torch
import torch.nn as nn
from queue import PriorityQueue
from copy import deepcopy
import operator
from pytorch_pretrained_bert import BertModel
import adamod
from utils.log import timeit
# import torch.nn.functional as F
# from sklearn.metrics import f1_score
from torch import optim
# from detector_utils import masked_cross_entropy
from utils.io_utils import masked_cross_entropy, top_k_top_p_filtering


class KGIRNet(nn.Module):
    """
    Sequence to sequence model with Attention
    """
    def __init__(self, hidden_size, max_r, src_vocab, trg_vocab, decoder_lr_ration, sos_tok, eos_tok, emb_dim,
                 gpu=False, lr=0.001, n_layers=1, clip=10.0, tot_rel=10, tot_ent=256, bert_hidden=768, itoe=None,
                 dropout=0.1, rnn_dropout=0.3, emb_drop=0.5, teacher_forcing_ratio=5.0, pretrained_emb_dec=None):
        super(KGIRNet, self).__init__()
        self.name = "KGIRNet_bert"
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.hidden_size = hidden_size
        self.bert_hidden = bert_hidden
        self.tot_rel = tot_rel
        self.itoe = itoe
        self.max_r = max_r  # max response len
        self.lr = lr
        self.tot_ent = tot_ent
        self.top_p = 10.0
        self.top_k = 1
        self.emb_dim = emb_dim
        self.decoder_learning_ratio = decoder_lr_ration
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_drop = rnn_dropout
        self.emb_drop = emb_drop
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_tok = sos_tok
        self.eos_tok = eos_tok
        self.clip = clip
        self.use_cuda = gpu
        self.ent_loss = nn.CrossEntropyLoss()

        # Use single RNN for both encoder and decoder
        # self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        # initializing the model
        self.encoder = EncoderRNN(n_layers=self.n_layers, emb_size=self.bert_hidden, dropout=self.dropout,
                                  hidden_size=self.hidden_size, out_rel=self.tot_rel,
                                 gpu=self.use_cuda)
        self.decoder = Decoder(hidden_size=self.hidden_size, emb_dim=self.emb_dim, vocab_size=self.trg_vocab,
                               dropout=self.rnn_drop, pretrained_emb=pretrained_emb_dec)
        # self.ent_predictor = EntityPredictor(emb_dim=self.emb_dim, n_vocab=self.output_size, gpu=self.use_cuda,
        #                                  pretrained_emb=pretrained_emb, emb_drop=self.emb_drop, ent_size=self.tot_ent)

        # nn.init.xavier_normal_(self.kg_trainer.weight)
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            # self.ent_predictor = self.ent_predictor.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        # self.encoder_optimizer = adamod.AdaMod(self.encoder.parameters(), lr=lr, beta3=0.999)
        # self.enttity_loss =
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)
        # self.decoder_optimizer = adamod.AdaMod(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio, beta3=0.999)
        # self.ent_pred_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.loss = 0
        self.print_every = 1

    # def train_batch(self, input_batch, input_chunk, out_batch, input_mask, target_mask, kb, kb_mask, sentient_orig):
    def train_batch(self, input_batch, input_mask, token_type, input_graph, response_out, response_mask, input_ent):
        # input graph = the laplacian of the subgraph B X V
        self.encoder.train(True)
        self.decoder.train(True)
        # self.ent_predictor.train(True)
        # self.kg_trainer.train(True)
        b_size = input_batch.size(0)
        # print (b_size)
        # print (self.kb_max_size)
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # self.ent_pred_optimizer.zero_grad()

        # loss_Vocab,loss_Ptr,loss_Gate = 0,0,0
        # Run words through encoder

        encoder_outputs, encoder_hidden, entity_pred = self.encoder(input_batch, input_mask, token_type)
        # ent_pred = self.ent_predictor(input_batch)
        response_out = response_out.transpose(0, 1)  # B X S --> S X B
        target_len = response_out.size(0)  # Get S
        # print (min(max(target_len), self.max_r))
        max_target_length = min(target_len, self.max_r)
        # print (max_target_length)
        if not isinstance(max_target_length, int):
            max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())

        # Prepare input and output variables
        decoder_input = torch.tensor([self.sos_tok] * b_size).long()
        # kg_out_seq = torch.zeros(int(max_target_length), b_size, self.kb_max_size)
        all_decoder_outputs_vocab = torch.zeros(int(max_target_length), b_size, self.trg_vocab)

        if self.use_cuda:
            decoder_input = decoder_input.cuda()
            # kg_out_seq = torch.zeros(int(max_target_length), b_size, self.kb_max_size).cuda()
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
        # else:
            # only_vocab_mask = torch.ones(b_size, self.output_size) # mask to remove all entities not in input graph

        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        use_teacher_forcing = random.randint(0, 10) < self.teacher_forcing_ratio

        if use_teacher_forcing:
            for t in range(max_target_length):
                decoder_vocab, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)  # B X H & TUPLE
                decoder_vocab = decoder_vocab
                all_decoder_outputs_vocab[t] = decoder_vocab
                decoder_input = response_out[t].long()  # Next input is current target
        else:
            # print ('Not TF..')
            for t in range(max_target_length):
                # inp_emb_d = self.embedding(decoder_input)
                decoder_vocab, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs, input_mask)
                decoder_vocab = decoder_vocab
                all_decoder_outputs_vocab[t] = decoder_vocab
                topv, topi = decoder_vocab.data.topk(1)  # get prediction from decoder
                decoder_input = topi.view(-1)  # use this in the next time-steps

        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> B x S X VOCAB
            response_out.transpose(0, 1).contiguous(),  # -> B x S
            response_mask
        )


        ent_loss = self.ent_loss(entity_pred, input_ent)
        loss = loss_Vocab + ent_loss
        loss.backward()
        # ent_loss.backward()

        # ent_loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        #  Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # self.ent_pred_optimizer.step()
        self.loss += loss.item()

    def evaluate_batch(self, input_batch, input_mask, token_type, input_q_text, input_ent, get_graph_laplacian,
                       input_graph, evaluating=True, beam_width=10):
        """
        evaluating batch
        :param input_batch:
        :param input_graph: B X V
        :param out_batch:
        :param input_mask:
        :param target_mask:
        :return:
        """
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        encoder_outputs, encoder_hidden, entity_pred = self.encoder(input_batch, input_mask, token_type)
        # ent_pred = self.ent_predictor(input_batch)
        b_size = input_batch.size(0)
        pred_entities = torch.argmax(entity_pred, dim=-1)
        # target_len = torch.sum(target_mask, dim=0)
        # target_len = response_out.size(1)
        # response_out = response_out.transpose(0, 1)  # B X S --> S X B
        # print (min(max(target_len), self.max_r))
        # max_target_length = (min(target_len, self.max_r))
        # print (max_target_length)
        # if not isinstance(max_target_length, int):
        #     max_target_length = int(max_target_length.cpu().numpy()) if self.use_cuda else int(max_target_length.numpy())
        # Get output mask
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers], encoder_hidden[1][:self.decoder.n_layers])
        if evaluating:
            decoded_words = self.beam_decode(batch_size=b_size, decoder_hiddens=decoder_hidden, input_kgs=input_graph,
                                             encoder_outputs=encoder_outputs, input_masks=input_mask, beam_width=beam_width)
        else:
            pred_ent_text = [self.itoe[e.item()] for e in pred_entities]
            # print (pred_ent_text, input_ent)
            pred_i_g = [get_graph_laplacian(e, input_q_text[j]) for j, e in enumerate(pred_ent_text)]
            pred_i_g = torch.stack(pred_i_g, dim=0)
            if self.use_cuda:
                pred_i_g = pred_i_g.cuda()
            # Prepare input and output va riables
            # decoder_input = torch.tensor([self.sos_tok] * b_size).long()
            # decoded_words = torch.zeros(int(max_target_length), b_size)
            # all_decoder_outputs_vocab = torch.zeros(int(max_target_length), b_size, self.trg_vocab)
            # if self.use_cuda:
                # decoder_input = decoder_input.cuda().long()
                # all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
                # decoded_words = decoded_words.cuda()
            decoded_words = self.beam_decode(batch_size=b_size, decoder_hiddens=decoder_hidden, input_kgs=pred_i_g,
                                             encoder_outputs=encoder_outputs, input_masks=input_mask, beam_width=beam_width)

        # self.ent_predictor.train(True)
        # self.embedding.train(True)

        return decoded_words, pred_entities
    @timeit
    def beam_decode(self, batch_size, decoder_hiddens, input_kgs,encoder_outputs=None, input_masks=None, beam_width=10):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [B, T, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        # target_tensor = target_tensor.permute(1, 0)
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(batch_size):  # batch_size
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[idx, :, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)  # [B,T,H]=>[1,T,H]=>[1,T,H]
            input_mask = input_masks[idx, :]
            input_kg = input_kgs[idx, :]

            # Start with the start of the sentence token
            if self.use_cuda:
                decoder_input = torch.Tensor([self.sos_tok]).long().cuda()
            else:
                decoder_input = torch.Tensor([self.sos_tok]).long()

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                # print('--best node seqs len {} '.format(n.leng))
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == self.eos_tok and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output, input_mask)
                decoder_output = decoder_output * input_kg
                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)


class EncoderRNN(nn.Module):
    """
    Encoder RNN module
    """
    def __init__(self, emb_size, hidden_size, out_rel, n_layers=1, dropout=0.1, emb_drop=0.2,
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
        # self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.entity_classifier = nn.Linear(emb_size, out_rel)
        # self.bert.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(emb_drop)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, dropout=self.dropout, batch_first=True)
        if pretrained_emb is not None:
           self.embedding.weight.data.copy_(pretrained_emb)
        if train_emb == False:
           self.embedding.weight.requires_grad = False
        # self.rnn = rnn

    def init_weights(self, b_size):
        # intiialize hidden weights
        c0 = torch.rand(self.n_layers, b_size, self.hidden_size)
        h0 = torch.rand(self.n_layers, b_size, self.hidden_size)

        if self.gpu:
            c0 = c0.cuda()
            h0 = h0.cuda()

        return h0, c0

    def forward(self, inp_q, input_mask, token_types, input_lengths=None):
        '''
        :param inp_q: B X S X E
        :param input_mask: B X S
        :param input_lengths:
        :return: encode input query
        '''
        # input_q =numpy S X B input_mask = S X B
        # embeddeinputs, inp_mask, encoder_hidden, decoder_out, kb, kb_mask, last_sentientd = self.embedding(input_q)
        embedded, pooled = self.bert(input_ids=inp_q, attention_mask=input_mask, token_type_ids=token_types,
                                     output_all_encoded_layers=False)  # B X S X E
        embedded = self.embedding_dropout(embedded) # B X S X E
        # embedded = self.embedding_dropout(inp_q) # B X S X E
        b_size = inp_q.size(0)
        hidden = self.init_weights(b_size)
        embedded = self.embedding_dropout(embedded)

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)

        outputs, hidden = self.rnn(embedded, hidden)  # outputs = B X S X n_layers*H, hidden = 2 * [B X 1 X H]
        outputs = outputs * input_mask.unsqueeze(-1)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)

        # Predict entity
        entity_pred = self.entity_classifier(pooled)
        return outputs, hidden, entity_pred


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

        seq_len = encoder_outputs.size(1)  # get sequence lengths S
        H = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1)   # B X S X H
        energy = torch.tanh(self.W_h(torch.cat([H, encoder_outputs], -1))) # B X S X H
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
    def __init__(self, hidden_size, emb_dim, vocab_size, n_layers=1, dropout=0.2,
                 pretrained_emb=None, train_emb=True, emb_drop=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_emb is not None:
           self.embedding.weight.data.copy_(pretrained_emb)
        if train_emb == False:
           self.embedding.weight.requires_grad = False
        self.embedding_drop = nn.Dropout(emb_drop)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_size, n_layers, dropout=dropout)
        # self.rnn = rnn
        # self.out = nn.Linear(self.hidden_size, vocab_size)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            self.dropout,
            nn.Linear(self.hidden_size*2, vocab_size)
        )
        # self.concat = nn.Linear(hidden_size * 2, hidden_size*2)
        # self.out = nn.Linear(hidden_size * 2, vocab_size)
        # Attention
        self.attention = Attention(hidden_size, hidden_size)
        self.out.apply(self.init_weights)
        # nn.init.xavier_normal_(self.concat.weight)
        # nn.init.xavier_normal_(self.out.weight)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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
        embedded = self.embedding_drop(embedded)
        # print (embedded.size())

        embedded = embedded.view(1, batch_size, self.emb_dim)  # S=1 x B x N
        # print (embedded.size())
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
        # concat_output = torch.tanh(self.concat(concat_input))
        # print (concat_output.size())
        # Finally predict next token (Luong eq. 6, without softmax)
        # output = self.out(self.concat(concat_input))
        output = self.out(concat_input)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, alpha


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # beam-search

    def __lt__(self, other):
        return self.leng < other.leng  #

    def __gt__(self, other):
        return self.leng > other.leng