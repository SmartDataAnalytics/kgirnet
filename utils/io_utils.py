import torch
import torch.nn as nn
import os


def masked_cross_entropy(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    # mask = mask.transpose(0, 1).float()
    length = torch.sum(mask, dim=-1)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))  # -1 means inferred from other dimensions
    # print (logits_flat)
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log_softmax(logits_flat, dim=-1)
    # print (log_probs_flat)
    # target_flat: (batch * max_len, 1)
    if target.size(0) == 1:
        target_flat = target.transpose(0, 1).long()
    else:
        target_flat = target.view(-1, 1).long()

    # losses_flat: (batch * max_len, 1)
    # print (target_flat.size(), log_probs_flat.size())
    # print (log_probs_flat.size(), target_flat.size())
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    # print (logits.float().sum())
    losses = losses * mask
    loss = losses.sum() / (length.float().sum() + 1e-10)
    return loss


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    log_np = logits.detach().numpy()
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        indices_to_remove_np = indices_to_remove.numpy()

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        cum_prob = cumulative_probs.detach().numpy()
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_remove_np = indices_to_remove.numpy()
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    log_np_removed = logits.detach().numpy()
    return logits


def save_model(model, name):
    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')

    torch.save(model.state_dict(), 'saved_models/{}.bin'.format(name))


def load_model(model, name, gpu=True):
    if gpu:
        model.load_state_dict(torch.load('saved_models/{}.bin'.format(name)))
    else:
        model.load_state_dict(torch.load('saved_models/{}.bin'.format(name), map_location=lambda storage, loc: storage))

    return model