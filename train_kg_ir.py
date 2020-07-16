# from models.KG_IR_Net_vocab import KGIRNet
from utils.args import get_args
import numpy as np
import torch
from utils.perf_utils import compute_f1, get_f1
from utils.decoder_utils import DecodeSentences
from tqdm import tqdm
import pandas as pd
# import random
# from sklearn.metrics import accuracy_score
from utils.io_utils import save_model, load_model
# from metrics import EmbeddingMetrics
from evaluators.bleu import get_moses_multi_bleu
# from evaluators.eval_WE_WPI_multi import
# from evaluators.gleu import corpus_gleu
# from nltk.translate.meteor_score import meteor_score

# Get arguments
args = get_args()
print (args)
# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
wewpi_eval=0.0 # initialize with int and only once
fasttext_emb='/home/debanjan/submission_soccer/data/wiki.simple.bin'
# fasttext_emb='data/wiki.simple.bin'
if args.gpu:
    torch.cuda.manual_seed(args.randseed)
    # fasttext_emb='/data/home1/dchaudhuri/deep/soccerbot_acl/vocab/cc.en.300.bin'
    fasttext_emb='/home/dchaudhuri/deep/submission_soccer/data/wiki.simple.bin'


# Define variables
if args.dataset == 'incar':
    data_path = 'data/incar/'
    model_name = 'IR_KG_Net_incar_sep_dec'
    test_results = 'test_predicted_ir_kg_net_incar'
else:
    data_path = 'data/soccer/'
    model_name = 'IR_KG_Net_soccer'
    test_results = 'test_predicted_ir_kg_net_soccer'


if args.use_bert:
    # make imports
    from models.KG_IR_Net_bert import KGIRNet
    if args.dataset == 'incar':
        # from models.KG_IR_Net_bert_incar import KGIRNet
        from utils.batcher.incar_batcher_sep_vocab_bert import InCarBatcher

        chat_data = InCarBatcher(data_path=data_path, fasttext_model=fasttext_emb, max_sent_len=args.max_sent_len,
                                 batch_size=args.batch_size, gpu=args.gpu, max_resp_len=args.resp_len, domain=args.dataset)
    else:
        # from models.KG_IR_Net_bert_incar import KGIRNet
        from utils.batcher.soccer_batcher_sep_vocab_bert import SoccerBatcher

        chat_data = SoccerBatcher(data_path=data_path, fasttext_model=fasttext_emb, max_sent_len=args.max_sent_len, min_vocab_freq=0.0,
                                 batch_size=args.batch_size, gpu=args.gpu, max_resp_len=args.resp_len, domain=args.dataset)
    model = KGIRNet(hidden_size=args.hidden_size, max_r=args.resp_len, gpu=args.gpu, emb_dim=args.words_dim, tot_rel=len(chat_data.itoe),
                    trg_vocab=len(chat_data.trg_vocab), src_vocab=len(chat_data.src_vocab),
                    lr=args.lr, dropout=args.rnn_dropout, emb_drop=args.emb_drop,
                    teacher_forcing_ratio=args.teacher_forcing, itoe=chat_data.itoe,
                    sos_tok=chat_data.trg_stoi[args.sos_tok], tot_ent=len(chat_data.etoi), decoder_lr_ration=5.0,
                    eos_tok=chat_data.trg_stoi[args.eos_tok])
    model_name = model_name+'_bert'
    test_results = test_results+'_bert.csv'
else:
    from models.KG_IR_Net_vocab import KGIRNet
    from utils.batcher.incar_batcher_sep_vocab import InCarBatcher

    # Batchers
    chat_data = InCarBatcher(data_path=data_path, max_sent_len=20, batch_size=args.batch_size, gpu=args.gpu,
                             fasttext_model=fasttext_emb, max_resp_len=args.resp_len, domain=args.dataset)
    # Get model
    model = KGIRNet(hidden_size=args.hidden_size, max_r=args.resp_len, gpu=args.gpu, emb_dim=args.words_dim, src_vocab=len(chat_data.src_stoi),
                    trg_vocab=len(chat_data.trg_vocab),lr=args.lr, dropout=args.rnn_dropout, emb_drop=args.emb_drop, teacher_forcing_ratio=args.teacher_forcing,
                    sos_tok=chat_data.trg_stoi[args.sos_tok], eos_tok=chat_data.trg_stoi[args.eos_tok],tot_ent=len(chat_data.etoi), decoder_lr_ration=args.decoder_lr,
                    # pretrained_emb=ent_det.word_embed.weight.data)
                    pretrained_emb=torch.from_numpy(chat_data.src_vectors.astype(np.float32)),
                    pretrained_emb_dec=torch.from_numpy(chat_data.trg_vectors.astype(np.float32)))
    # model_name = 'IR_KG_Net_incar_sep_dec'
    test_results = test_results+'.csv'
# metrics = EmbeddingMetrics(embeddig_dict=chat_data.vocab_glove)

test_out = pd.DataFrame()
decoder_utils = DecodeSentences(chat_data, data_path=data_path, domain=args.dataset)


def train():

    best_sc = 0.0
    f1_sc = 0.0
    for epoch in range(args.epochs):
        epsilon = 0.000000001
        model.train()
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        train_iter = enumerate(chat_data.get_iter('train', domain=args.dataset))
        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            train_iter.total = chat_data.n_train // chat_data.batch_size
        for it, mb in train_iter:
            if args.use_bert:
                q, q_m, t_t, i_g, r, r_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg = mb
                model.train_batch(q, q_m, t_t, i_g, r, r_m, i_e)
            else:
                q, q_m, i_g, r, r_m, i_e, t_r, c_e, i_q, o_r, c_o, l_kg = mb
                model.train_batch(q, q_m, i_g, r, r_m, i_e)
            train_iter.set_description(model.print_loss())

        print('\n\n-------------------------------------------')
        print('Validation')
        print('-------------------------------------------')
        val_iter = enumerate(chat_data.get_iter('val', domain=args.dataset))

        if not args.no_tqdm:
            val_iter = tqdm(val_iter)
            val_iter.set_description_str('Validation')
            val_iter.total = chat_data.n_val // chat_data.batch_size

        val_loss = 0.0
        predicted_s = []
        orig_s = []
        gold_ent = []
        pred_ent = []
        ent_acc = []
        local_kg = []
        n_iter = 0
        for it, mb in val_iter:
            # q, q_m, i_g, r, r_m, i_e, t_r, c_e, i_q, o_r, c_o, l_kg = mb
            if args.use_bert:
                q, q_m, t_t, i_g, r, r_m, i_e, t_r, c_e, i_q, o_r, c_o, l_kg = mb
                pred, pred_entities = model.evaluate_batch(q, q_m, t_t, i_q, i_e, decoder_utils.get_graph_lap, i_g,
                                                           evaluating=False)
            else:
                q, q_m, i_g, r, r_m, i_e, t_r, c_e, i_q, o_r, c_o, l_kg = mb
                pred = model.evaluate_batch(q, q_m, i_g, r, r_m, i_e)
            s_g = t_r
            s_p, obj_pred, rel_pred, predicted_orig, kg_l = decoder_utils.get_sentences(pred, i_e, l_kg)
            local_kg.append(kg_l)
            predicted_s.append(s_p)
            orig_s.append(s_g)
            pred_ent.append(obj_pred)
            gold_ent.append(c_o)
            n_iter+=1
        print('\n\n-------------------------------------------')
        print ('Sample prediction')
        print('-------------------------------------------')
        for k, o in enumerate(s_g):
           print ('Original:' + o)
           try:
               print ('Predicted:' + s_p[k])
           except UnicodeEncodeError:
               print ('Predicted: '.format(s_p[k]))
        print('-------------------------------------------')

        # flatten the lists
        predicted_s = [q for ques in predicted_s for q in ques]
        orig_s = [q for ques in orig_s for q in ques]
        gold_ent = [g for gold in gold_ent for g in gold]
        pred_ent = [p for pred in pred_ent for p in pred]
        local_kg = [l for l_k in local_kg for l in l_k]

        m2s_f1 = compute_f1(gold_ent, predicted_s, local_kg)
        # f1_scores = [get_f1(g_ent, pred_ent[j]) for j, g_ent in enumerate(gold_ent) if '' not in g_ent]
        if args.dataset == 'soccer':
            f1_scores = [get_f1(g_ent, pred_ent[j]) for j, g_ent in enumerate(gold_ent) if '' not in g_ent]
        else:
            f1_scores = [get_f1(g_ent, pred_ent[j]) for j, g_ent in enumerate(gold_ent) if g_ent]
        f1_ent = np.average(f1_scores)

        # Get BLEU scores
        moses_bleu, bleu1, bleu2, bleu3, bleu4 = get_moses_multi_bleu(predicted_s, orig_s, lowercase=True)

        print ('Length of pred:' + str(len(orig_s)) + ' moses bleu: '+ str(moses_bleu))
        print("Entity F1-score (mem2seq): ", m2s_f1)
        print("Entity F1-score (ours): ", f1_ent)
        # print("Entity accuracy: ", np.average(ent_acc))
        # if moses_bleu > best_sc:
        if f1_ent > f1_sc:
            f1_sc = f1_ent
            best_sc = moses_bleu
            print('Saving best model')
            print('Moses Bleu score:{:.4f}, F1:{:.4f}'.format(best_sc, f1_sc))
            save_model(model, model_name)
        else:
            print ('Not saving the model. Best validation BLEU so far:{:.4f} with F1 (ours):{:.4f}'.format(best_sc, f1_sc))
        # print ('Validation Loss:{:.2f}'.format(val_loss/val_iter.total))


def _test(model, k=10):
    # global wewpi_eval
    model = load_model(model, model_name, args.gpu)
    print('\n\n-------------------------------------------')
    print('Testing')
    print('-------------------------------------------')
    test_iter = enumerate(chat_data.get_iter('test', domain=args.dataset))
    if not args.no_tqdm:
        test_iter = tqdm(test_iter)
        test_iter.set_description_str('Testing')
        test_iter.total = chat_data.n_test // chat_data.batch_size
    # test_loss = 0.0

    predicted_s = []
    orig_s = []
    gold_ent = []
    local_kg = []
    pred_ent = []
    pred_s_orig = []
    input_q = []
    all_predictions_topk = []
    n_iter = 0
    for it, mb in test_iter:
        all_preds = []
        if args.use_bert:
            q, q_m, t_t, i_g, r, r_m, i_e, t_r, c_e, i_q, o_r, c_o, l_kg = mb
        else:
            q, q_m, i_g, r, r_m, i_e, t_r, c_e, i_q, o_r, c_o, l_kg = mb
        input_q.append(i_q)
        if args.use_bert:
            if args.dataset == 'incar':
                pred, pred_entities = model.evaluate_batch(q, q_m, t_t, i_q, i_e, decoder_utils.get_graph_lap, i_g, evaluating=False, beam_width=k)
            else:
                pred, pred_entities = model.evaluate_batch(q, q_m, t_t, i_q, i_e, decoder_utils.get_graph_lap, i_g, evaluating=False, beam_width=k)
        else:
            # i_e =
            pred_entities = [chat_data.itoe[e.item()] for e in i_e]
            pred = model.evaluate_batch(q, q_m, i_g, r, r_m, i_e)

        s_p, obj_pred, rel_pred, predicted_orig, kg_l = decoder_utils.get_sentences(pred, pred_entities, l_kg)
        predicted_s.append(s_p)
        local_kg.append(kg_l)
        orig_s.append(t_r)
        pred_s_orig.append(predicted_orig)
        pred_ent.append(obj_pred)
        gold_ent.append(c_o)
        all_predictions_topk.append(all_preds)
        n_iter += 1
    print('\n\n-------------------------------------------')
    print('Sample prediction')
    print('-------------------------------------------')
    for k, o in enumerate(t_r):
        print('Original:' + o)
        try:
            print('Predicted:' + s_p[k])
        except UnicodeEncodeError:
            print('Predicted: '.format(s_p[k]))
    print('-------------------------------------------')

    predicted_s = [q for ques in predicted_s for q in ques]
    orig_s = [q for ques in orig_s for q in ques]
    input_q = [q for ques in input_q for q in ques]
    pred_s_orig = [q for ques in pred_s_orig for q in ques]
    local_kg = [l for l_k in local_kg for l in l_k]
    gold_ent = [g for gold in gold_ent for g in gold]
    all_predictions_topk = [topk for top in all_predictions_topk for topk in top]
    out_top_k = (input_q, orig_s, all_predictions_topk)
    pred_ent = [p for pred in pred_ent for p in pred]

    # pred_ent = [[w.replace('<ent>', '') for w in pred_sent.split() if '<ent>' in w] for pred_sent in predicted_s]
    if args.dataset == 'soccer':
        f1_scores = [get_f1(g_ent, pred_ent[j]) for j, g_ent in enumerate(gold_ent) if '' not in g_ent]
    else:
        f1_scores = [get_f1(g_ent, pred_ent[j]) for j, g_ent in enumerate(gold_ent) if g_ent]
    # print (f1_scores)
    f1 = np.average(f1_scores)

    m2s_f1 = compute_f1(gold_ent, predicted_s, local_kg)

    moses_bleu, bleu1, bleu2, bleu3, bleu4 = get_moses_multi_bleu(predicted_s, orig_s, lowercase=True)
    print ('BLEU scores', bleu1, bleu2, bleu3, bleu4)
    print ("Moses Bleu:" + str(moses_bleu))
    print("F1 score (ours): ", f1)
    print("F1 score (mem2seq): ", m2s_f1)
    # print("WE_WPI score: ", we_wpi_score)
    test_out['Input_query'] = input_q
    test_out['original_response'] = orig_s
    test_out['predicted_response'] = predicted_s
    test_out['predicted_response_orig'] = pred_s_orig
    test_out['kv_entities'] = gold_ent
    test_out['predicted_ent'] = pred_ent
    print ('Saving the test predictions......')
    test_out.to_csv(test_results, index=False, sep='\t')
    np.save('topk_files.npy', out_top_k)


if __name__ == '__main__':
    train()
    # test_convo(model)
    _test(model)
