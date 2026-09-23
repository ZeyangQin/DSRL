import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn, dtype
from torch.nn.functional import cosine_similarity
from torch.nn.functional import cross_entropy
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers import PreTrainedModel, AutoModel
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

def avg_pooling(last_hidden, attention_mask):
    '''
    average of hidden states at each token.
    '''
    features = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    return features

def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class UCL_Loss:
    def __init__(self, temp):
        'UCL loss in textual representation space'

        self.loss_fct = nn.CrossEntropyLoss()
        self.temp = temp
        self.sim = Similarity(temp)

    def __call__(self, pos_features, pos_features_aug, neg_features):
        cos_pos = self.sim(pos_features.unsqueeze(1), pos_features_aug.unsqueeze(0))
        cos_neg = self.sim(pos_features.unsqueeze(1), neg_features.unsqueeze(0))
        cos_pos = torch.cat((cos_pos, cos_neg), dim=1)
        labels_con = torch.arange(cos_pos.size(0)).long().cuda()
        loss = self.loss_fct(cos_pos, labels_con)

        return loss


class RCL_Loss:
    def __init__(self, temp=None):
        'RCL loss in conditional textual representation space'
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def __call__(self, pos1, pos2, neg1, neg2, labels):
        dist_pos = self.cos(pos1, pos2)
        dist_neg = self.cos(neg1, neg2)

        dist_all = torch.cat([dist_pos, dist_neg], dim=0)
        diff = dist_all[:, None] - dist_all[None, :]
        labels = labels[:, None] - labels[None, :]

        loss = F.mse_loss(labels.float(), diff, reduction='mean')
        return loss


class Out_InfoNCE_Loss:
    def __init__(self, temp):
        'Contrastive loss in conditional textual representation space'

        self.loss_fct = nn.CrossEntropyLoss()
        self.temp = temp
        self.sim = Similarity(self.temp)

    def __call__(self, positives1, positives2, negatives1, negatives2, labels):
        cos_pos_data = self.sim(positives1, positives2)
        cos_neg_data = self.sim(negatives1, negatives2)
        cos_sim = torch.cat((cos_pos_data.unsqueeze(-1), cos_neg_data.unsqueeze(-1)), dim=1)
        labels_con = torch.zeros(cos_pos_data.size(0)).long().cuda()
        loss = self.loss_fct(cos_sim, labels_con)
        return loss


# Pooler class. Copied and adapted from SimCSE code
class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def extract_sentence_condition_features_cross(features, attention_mask_record, part=None):

    '''
    features: [bsz, sentence_condition_length, hidden_size]
    attention_mask_record: [bsz, sentence_condition_length]，padding(0)，s1(1); s2(2); c(3); [CLS](4)]; [SEP](5)


    return:
    sentence_features: [bsz, sentence_length, hidden_size]
    condition_features: [bsz, condition_length, hidden_size]
    CLS_features: [bsz, 3, hidden_size]
    SEP_features： [bsz, 3, hidden_size]
    sentence_attn_mask: [bsz, sentence_length]
    condition_attn_mask: [bsz, condition_length]
    '''

    batch_size = features.size(0)
    hidden_size = features.size(2)

    sentence_1_mask = (attention_mask_record == 1)
    sentence_2_mask = (attention_mask_record == 2)
    condition_mask = (attention_mask_record == 3)
    CLS_mask = (attention_mask_record == 4)
    SEP_mask = (attention_mask_record == 5)

    sentence_1_lengths = sentence_1_mask.sum(dim=1)
    sentence_2_lengths = sentence_2_mask.sum(dim=1)
    condition_lengths = condition_mask.sum(dim=1)
    CLS_length = CLS_mask.sum(dim=1)
    SEP_length = SEP_mask.sum(dim=1)

    max_sentence_1_len = max(sentence_1_lengths.max().item(), 11)
    max_sentence_2_len = max(sentence_2_lengths.max().item(), 11)
    max_condition_len = max(condition_lengths.max().item(), 11)
    max_CLS_len = CLS_length.max().item()
    max_SEP_len = SEP_length.max().item()

    sentence_1_features = torch.zeros(batch_size, max_sentence_1_len, hidden_size, device=features.device)
    sentence_2_features = torch.zeros(batch_size, max_sentence_2_len, hidden_size, device=features.device)
    condition_features = torch.zeros(batch_size, max_condition_len, hidden_size, device=features.device)
    CLS_features = torch.zeros(batch_size, max_CLS_len, hidden_size, device=features.device)  # [bsz, 3, hidden_size]
    SEP_features = torch.zeros(batch_size, max_SEP_len, hidden_size, device=features.device)  # [bsz, 3, hidden_size]


    sentence_1_attn_mask = torch.zeros(batch_size, max_sentence_1_len, dtype=torch.long, device=features.device)
    sentence_2_attn_mask = torch.zeros(batch_size, max_sentence_2_len, dtype=torch.long, device=features.device)
    condition_attn_mask = torch.zeros(batch_size, max_condition_len, dtype=torch.long, device=features.device)
    CLS_attn_mask = torch.zeros(batch_size, max_CLS_len, dtype=torch.long, device=features.device)  # [bsz, 3]
    SEP_attn_mask = torch.zeros(batch_size, max_SEP_len, dtype=torch.long, device=features.device)  # [bsz, 3]

    for i in range(batch_size):

        sent_1_mask_i = sentence_1_mask[i]
        sent_2_mask_i = sentence_2_mask[i]
        cond_mask_i = condition_mask[i]
        CLS_mask_i = CLS_mask[i]
        SEP_mask_i = SEP_mask[i]

        sentence_1_feat_i = features[i][sent_1_mask_i]
        sentence_2_feat_i = features[i][sent_2_mask_i]

        condition_feat_i = features[i][cond_mask_i]
        CLS_feat_i = features[i][CLS_mask_i]
        SEP_feat_i = features[i][SEP_mask_i]

        sentence_1_features[i, :len(sentence_1_feat_i)] = sentence_1_feat_i
        sentence_2_features[i, :len(sentence_2_feat_i)] = sentence_2_feat_i
        condition_features[i, :len(condition_feat_i)] = condition_feat_i
        CLS_features[i, :len(CLS_feat_i)] = CLS_feat_i
        SEP_features[i, :len(SEP_feat_i)] = SEP_feat_i

        sentence_1_attn_mask[i, :len(sentence_1_feat_i)] = 1
        sentence_2_attn_mask[i, :len(sentence_2_feat_i)] = 1
        condition_attn_mask[i, :len(condition_feat_i)] = 1
        CLS_attn_mask[i, :len(CLS_feat_i)] = 1
        SEP_attn_mask[i, :len(SEP_feat_i)] = 1

    return sentence_1_features, sentence_2_features, condition_features, CLS_features, SEP_features, sentence_1_attn_mask, sentence_2_attn_mask, condition_attn_mask, CLS_attn_mask, SEP_attn_mask


class CGA(nn.Module):
    def __init__(self, token_top_k):
        super(CGA, self).__init__()
        self.token_top_k = token_top_k

    def forward(
            self,
            sentence_last_hiddens,
            condition_last_hiddens,
            cls_last_hiddens,
            sentence_attn_mask,
            condition_attn_mask,
            cls_attn_mask,
    ):
        '''
        sentence_last_hiddens: [bsz, length, hidden_size]
        condition_last_hiddens: [bsz, length, hidden_size]
        cls_last_hiddens: [bsz, 3, hidden_size]
        sentence_attn_mask: [bsz, length]
        condition_attn_mask: [bsz, length]
        cls_attn_mask: [bsz, 3]
        '''
        bsz, length, hidden_size = sentence_last_hiddens.size()

        condition_features_avg_poling = avg_pooling(condition_last_hiddens, condition_attn_mask)
        # [bsz, 1, hidden_size]
        query = condition_features_avg_poling.unsqueeze(1)
        key = sentence_last_hiddens
        value = sentence_last_hiddens
        attention_scores = torch.matmul(query, key.transpose(1, 2))
        attention_scores = attention_scores / (hidden_size ** 0.5)
        sentence_mask_expanded = sentence_attn_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(sentence_mask_expanded == 0, float('-inf'))

        ################### top-k ########################
        topk_scores, topk_indices = torch.topk(attention_scores, k=self.token_top_k, dim=-1)
        core_new_scores = torch.full_like(attention_scores, float('-inf'))
        core_new_scores.scatter_(-1, topk_indices, topk_scores)
        ################### top-k ########################
        core_new_scores = F.softmax(core_new_scores, dim=-1)
        context = torch.matmul(core_new_scores, value)

        # [bsz, hidden_size]
        context = context.squeeze(1)

        return context

class CrossEncoderForClassification(PreTrainedModel):
    'Encoder model with backbone and classification head.'
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model
        classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
                )
        else:
            self.transform = None

        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False

        self.loss_out_infoNCE = Out_InfoNCE_Loss
        self.loss_out_infoNCE_kwargs = {'temp': 0.1}
        self.lambda_1 = config.lambda_1
        self.lambda_2 = config.lambda_2
        self.lambda_3 = config.lambda_3
        self.token_top_k = config.token_top_k
        self.cga_mechanism = CGA(token_top_k=self.token_top_k)
        self.loss_UCL = UCL_Loss
        self.loss_RCL = RCL_Loss
        self.loss_RCL_kwargs = {}
        self.loss_UCL_kwargs = {'temp': 0.1}

        self.in_transform = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.predict_head = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(6 * config.hidden_size, config.hidden_size),
            ACT2FN[config.hidden_act],
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            attention_mask_record=None,
            labels=None,
            **kwargs,
            ):
        bsz = input_ids.shape[0]
        input_ids_aug = input_ids.clone()
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        last_hiddens = outputs.last_hidden_state

        outputs_aug = self.backbone(
            input_ids=input_ids_aug,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        last_hiddens_aug = outputs_aug.last_hidden_state
        sentence_1_last_hiddens, sentence_2_last_hiddens, condition_last_hiddens, cls_last_hiddens, sep_last_hiddens, sentence_1_attn_mask, sentence_2_attn_mask, condition_attn_mask, cls_attn_mask, sep_attn_mask = extract_sentence_condition_features_cross(last_hiddens, attention_mask_record)
        sentence_1_last_hiddens_aug, sentence_2_last_hiddens_aug, condition_last_hiddens_aug, cls_last_hiddens_aug, sep_last_hiddens_aug, _,_,_,_,_ = extract_sentence_condition_features_cross(last_hiddens_aug, attention_mask_record)

        features_1 = self.cga_mechanism(sentence_1_last_hiddens, condition_last_hiddens, cls_last_hiddens, sentence_1_attn_mask, condition_attn_mask, cls_attn_mask)
        features_2 = self.cga_mechanism(sentence_2_last_hiddens, condition_last_hiddens, cls_last_hiddens, sentence_2_attn_mask, condition_attn_mask, cls_attn_mask)
        positives1, negatives1 = torch.split(features_1, bsz // 2, dim=0)
        positives2, negatives2 = torch.split(features_2, bsz // 2, dim=0)
        logits_cl = cosine_similarity(features_1, features_2, dim=1)
        loss = None

        # cls_flatten = cls_last_hiddens.reshape(bsz, -1)
        # sep_flatten = sep_last_hiddens.reshape(bsz, -1)
        # cls_sep_features = torch.cat((cls_flatten, sep_flatten), dim=-1)
        # logits_cls = self.predict_head(cls_sep_features).squeeze(1)

        logits = logits_cl  # self.lambda_2 * logits_cl + (1 - self.lambda_2) * logits_cls

        if labels is not None:
            pos_features, pos_features_aug, neg_features = avg_pooling(sentence_1_last_hiddens, sentence_1_attn_mask), avg_pooling(sentence_1_last_hiddens_aug, sentence_1_attn_mask), avg_pooling(sentence_2_last_hiddens, sentence_2_attn_mask)
            pos_features, pos_features_aug, neg_features = self.in_transform(pos_features), self.in_transform(pos_features_aug), self.in_transform(neg_features)
            loss = self.lambda_1 * self.loss_UCL(**self.loss_UCL_kwargs)(pos_features, pos_features_aug, neg_features)
            loss += self.loss_RCL(**self.loss_RCL_kwargs)(positives1, positives2, negatives1, negatives2, labels)
            loss += nn.MSELoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
