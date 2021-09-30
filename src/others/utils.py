import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import defaultdict

# from others.perplexity import read_sentences_from_file, BigramLanguageModel
# from others.perplexity import calculate_bigram_perplexity, calculate_unigram_perplexity

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def edit_distance2_backtracking(d, x, y, terminal_symbol):
    seq = list()
    edit_seqs = list()
    for _ in range(len(x)+2):
        edit_seqs.append([])
    if len(x) == 0:
        edit_seqs[0] = y
        return edit_seqs
    i = len(d) - 1
    j = len(d[0]) - 1
    while i >= 0 and j >= 0:
        if i == 0 and j == 0:
            break
        if j > 0 and d[i][j-1] < d[i][j]:
            seq.append((1, y[j-1]))
            j -= 1
        elif i > 0 and d[i-1][j] < d[i][j]:
            seq.append((2, x[i-1]))
            i -= 1
        else:
            seq.append((3, x[i-1]))
            i -= 1
            j -= 1
    prev_op = 0
    s = 0
    for k in range(len(seq)):
        op = seq[len(seq)-k-1][0]
        word = seq[len(seq)-k-1][1]
        if prev_op != 1:
            s += 1
        if op == 1:
            # insert
            edit_seqs[s-1].append(word)
        elif op == 2:
            # delete
            edit_seqs[len(x)+1].append(1)
        else:
            edit_seqs[len(x)+1].append(0)
        prev_op = op

    for k in range(len(edit_seqs)):
        if len(edit_seqs[k]) == 0:
            edit_seqs[k].append(terminal_symbol)
    return edit_seqs


def edit_distance2_with_dp(x, y):
    lx = len(x)
    ly = len(y)
    d = np.empty([lx+1, ly+1])
    for i in range(lx+1):
        d[i][0] = i
    for j in range(ly+1):
        d[0][j] = j
    for i in range(1, lx+1):
        for j in range(1, ly+1):
            d[i][j] = min(min(d[i-1][j], d[i][j-1]) + 1,
                          d[i-1][j-1] + 2 * (0 if x[i-1] == y[j-1] else 1))
    return d


def suggested_ed2_path(xs, ys, terminal_symbol):
    seq = list()
    for i in range(len(xs)):
        d = edit_distance2_with_dp(xs[i], ys[i])
        seq.append(
            edit_distance2_backtracking(
                d, xs[i], ys[i], terminal_symbol
            )
        )
    return seq


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def test_bleu(cand, ref):
    candidate = [line.strip() for line in open(cand, encoding='utf-8')]
    reference = [line.strip() for line in open(ref, encoding='utf-8')]
    if len(reference) != len(candidate):
        raise ValueError('The number of sentences in both files do not match.')
    if len(reference) == 0:
        return 0
    score = 0.
    for i in range(len(reference)):
        gold_list = reference[i].split()
        cand_list = candidate[i].split()
        score += sentence_bleu([gold_list], cand_list, smoothing_function=SmoothingFunction().method1)
    score /= len(reference)
    return score


def test_dist(cand):
    '''
        reference --- calc_diversity()
        https://github.com/microsoft/DialoGPT/blob/457835e7d8acd08acf7f6f0e980f36fd327ea37c/dstc/metrics.py#L131
    :param cand: filename
    :return: div1-unigram, div2-bigram
    '''
    tokens = [0.0, 0.0]
    types = [defaultdict(int),defaultdict(int)]
    for line in open(cand, encoding='utf-8'):
        words = line.strip('\n').strip().split()
        for n in range(2):
            for idx in range(len(words)-n):
                ngram = ' '.join(words[idx:idx+n+1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys())/tokens[0]
    if tokens[1] == 0:
        tokens[1] = 1
    div2 = len(types[1].keys())/tokens[1]
    return div1, div2


"""
def test_ppl(cand):
    '''
        score of perplexity
        接口：将insertion transformer model得到的word ins scores中每个sentence的score list保存在文本中，cand即表示该文本路径
    '''
    ppl = []
    for line in open(cand, encoding='utf-8'):
        word_probs = line.strip('\n').strip().split()
        if len(word_probs) == 0:
            continue
        sentence_prob_log_sum = 0.0
        n = len(word_probablities)
        for word_prob in word_probs:
            sentence_prob_log_sum += math.log(word_prob, 2)
        ppl.append(math.pow(2, -sentence_probability_log_sum * (1.0/n)))
    ave_score = np.mean(ppl)
    return ppl, ave_score
"""


def test_length(cand, ref, ratio=True):
    candidate = [len(line.split()) for line in open(cand, encoding='utf-8')]
    if len(candidate) == 0:
        return 0
    if ratio:
        reference = [len(line.split()) for line in open(ref, encoding='utf-8')]
        score = sum([candidate[i] / reference[i] for i in range(len(candidate))]) / len(candidate)
    else:
        score = sum(candidate) / len(candidate)
    return score


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    if x is None:
        return None
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.contiguous()\
         .view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def test_f1(acc_num, pred_num, gold_num):
    if gold_num == 0:
        r = 1
    else:
        r = acc_num / gold_num * 1.
    if pred_num == 0:
        p = 0
    else:
        p = acc_num / pred_num * 1.
    if p == 0. and r == 0.:
        f1 = -1
    else:
        f1 = (2 * p * r) / (p + r)
    return f1, p, r


"""
def rouge_results_to_str(results_dict):
    if results_dict is None:
        return "No Results.\n"
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100,
        results_dict["rouge_1_precision"] * 100,
        results_dict["rouge_2_precision"] * 100,
        results_dict["rouge_l_precision"] * 100
    )
"""


def rouge_results_to_str(results_dict):
    if results_dict is None:
        return "No Results.\n"
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/l): {:.2f}/{:.2f}/{:.2f}".format(
        results_dict["rouge-1"]['f'] * 100,
        results_dict["rouge-2"]['f'] * 100,
        results_dict["rouge-l"]['f'] * 100,
        results_dict["rouge-1"]['r'] * 100,
        results_dict["rouge-2"]['r'] * 100,
        results_dict["rouge-l"]['r'] * 100,
        results_dict["rouge-1"]['p'] * 100,
        results_dict["rouge-2"]['p'] * 100,
        results_dict["rouge-l"]['p'] * 100
    )
