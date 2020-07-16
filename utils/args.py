from argparse import ArgumentParser
import logging
import sys


def get_args():
    parser = ArgumentParser(description="Soccer chatbot")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--unk_tok', type=str, default='<unk>')
    parser.add_argument('--sos_tok', type=str, default='<sos>')
    parser.add_argument('--eos_tok', type=str, default='<eos>')
    parser.add_argument('--no_ent_tok', type=str, default='<no_ent>')
    parser.add_argument('--rel_tok', type=str, default='<rel>')
    # parser.add_argument('--mem_tok', type=str, default='<mem_ele>')
    parser.add_argument('--eou_tok', type=str, default='<eou>')
    parser.add_argument('--pad_tok', type=str, default='<pad>')
    parser.add_argument('--bert_cls', type=str, default='[CLS]')
    parser.add_argument('--bert_sep', type=str, default='[SEP]')
    parser.add_argument('--use_bert', type=bool, default=True)
    # parser.add_argument('--domain', type=str, default='incar')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resp_len', type=int, default=15)
    parser.add_argument('--max_sent_len', type=int, default=20)
    parser.add_argument('--inp_len', type=int, default=10)
    parser.add_argument('--dataset', type=str, default="incar")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decoder_lr', type=float, default=10)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dev_every', type=int, default=2000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--specify_prefix', type=str, default='id1')
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--input_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--stoi', type=str, default="vocab/w2i.npy")
    parser.add_argument('--vocab_glove', type=str, default="vocab/glove300.npy")
    parser.add_argument('--weight_decay',type=float, default=0)
    parser.add_argument('--teacher_forcing',type=int, default=10)
    parser.add_argument('--fix_embed', action='store_false', dest='train_embed')
    parser.add_argument('--hits', type=int, default=100)
    parser.add_argument('--no_tqdm', default=False, action='store_true', help='disable tqdm progress bar')
    parser.add_argument('--randseed', type=int, default=666, metavar='', help='random seed (default: 666)')
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='preproc_files/')
    parser.add_argument('--results_path', type=str, default='query_text')
    parser.add_argument('--emb_drop', type=float, default=0.3)
    parser.add_argument('--embpath',type=str,default='data/wiki.simple.bin')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


def getLogger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ])
    logger = logging.getLogger()
    return logger
