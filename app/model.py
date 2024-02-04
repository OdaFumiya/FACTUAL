# Author: Robert Guthrie

from tqdm import tqdm
import torch
import torch.nn as nn
import gensim
from utils import START_TAG,STOP_TAG,DEVICE,log_sum_exp,argmax,delete_zero

torch.manual_seed(1)



class BiLSTM_CRF(nn.Module):
    # 引数は辞書のサイズと埋め込み次元、隠れ層の次元
    def __init__(self, vocab_size, tag_to_ix, hidden_dim, w2vmodelpath):
        super(BiLSTM_CRF, self).__init__()
        model_dir = w2vmodelpath
        model = gensim.models.Word2Vec.load(model_dir)
        word_vectors = model.wv
        weights = word_vectors.vectors

        type(weights)  # numpy.ndarray
        weights.shape  # (3000000, 300)
        vocab_size = weights.shape[0]
        embedding_dim = weights.shape[1]
        embed = nn.Embedding(vocab_size, embedding_dim).to(DEVICE)
        embed.weight = nn.Parameter(torch.from_numpy(weights))

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim).to(DEVICE)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True).to(DEVICE)

        # LSTMの出力をタグ空間にマップする。
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 遷移パラメータの行列。 エントリ i,j は j から i に遷移するときのスコアである。
        # 
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # この2つの文は、「開始タグに転送しない」「停止タグから転送しない」という制約を強いています。
        #
        self.transitions.data[tag_to_ix[START_TAG], :] = -100000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -100000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        tmp=(torch.randn(2, 1, self.hidden_dim // 2).to(DEVICE),
                torch.randn(2, 1, self.hidden_dim // 2).to(DEVICE))      
        return tmp
    def _forward_alg(self, feats):
        # 大きさが1×k、値は全て-10000の行列を用意
        init_alphas = torch.full((1, self.tagset_size), -1000.).to(DEVICE)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # featの中のnext_tagに該当する値を1×kの行列に引き延ばす（値は同じ）
                # feat[next_tag] = tensor(0.3447)
                # emit_score = tensor([[ 0.3447,  0.3447,  0.3447,  0.3447,  0.3447]])
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                
                # タグからタグへの遷移行列（論文中のA）から注目しているnext_tagへの遷移スコアを取得
                # trans_scoreのi番目はiのタグからnext_tagへの遷移スコア
                trans_score = self.transitions[next_tag].view(1, -1)
                
                # next_tag_varのi番目はiのタグからnext_tagへ遷移するスコア
                # featのfor文の2週目以降（2つ目の単語以降）はそれまでの単語が算出したforwardが効いてくる
                next_tag_var = forward_var + trans_score + emit_score
                # next_tagへ遷移するスコアをlog_sum_expの合計で計算
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # それぞれのタグへの遷移スコア
            # 2つ目の単語以降はそれまでの単語がどのタグに遷移しやすいかのスコアが効いてくる
            forward_var = torch.cat(alphas_t).view(1, -1)
        # STOPへの遷移スコアを加える
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 返り値alphaはそれぞれの単語ごとのタグの遷移に関して、確信度が高く予想できれば小さい値、
        # 逆に曖昧な予想が多ければ大きい値になる
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # 与えられたタグ列のスコアを表示する
        score = torch.zeros(1).to(DEVICE)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(DEVICE), tags]).to(DEVICE)
        
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # ビタビ変数の対数空間での初期化
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(DEVICE)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # ステップiのforward_varは、ステップi-1のビタビ変数を保持する。
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
                          # このステップのバックポインタを保持します。
            viterbivars_t = []  # holds the viterbi variables for this step
                                # このステップのビタビ変数を保持する
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    def neg_log_likelihood(self, dataloader):
        forward_score = []
        gold_score = []
        for s,t in tqdm((torch.stack(dataloader,dim=1)), desc="learning:req", leave=False):
          sentence=delete_zero(s).to(DEVICE)
          tags=delete_zero(t).to(DEVICE)
          feats = self._get_lstm_features(sentence)
          forward_score.append(self._forward_alg(feats))
          gold_score.append(self._score_sentence(feats, tags))
        forward_score = torch.stack(forward_score, dim = 0)
        gold_score = torch.stack(gold_score, dim = 0)
        return forward_score.mean() - gold_score.mean()

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
