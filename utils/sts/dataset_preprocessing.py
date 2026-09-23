import random

def tokenize(sentences, conditions, tokenizer, total_len):
    attention_mask_record = []
    for i in range(len(sentences)):
        s = tokenizer.encode(sentences[i])
        c = tokenizer.encode(conditions[i])
        len_s, len_c, len_padding = len(s), len(c), total_len - len(s) - len(c)
        attention_mask_record.append(len_s * [1] + len_c * [2] + len_padding * [0])


    return attention_mask_record


def tokenize_decoupling(sentences_1, sentences_2, conditions, tokenizer):
    '''
    tokenize and construct attention_mask_record
    '''
    input_ids = []
    attention_mask_record, attention_mask = [], []
    for i in range(len(sentences_1)):
        s_1 = tokenizer.encode(sentences_1[i], add_special_tokens=True)
        bos, eos = s_1[0], s_1[-1]
        s_1 = tokenizer.encode(sentences_1[i], add_special_tokens=False)
        s_2 = tokenizer.encode(sentences_2[i], add_special_tokens=False)
        c = tokenizer.encode(conditions[i], add_special_tokens=False)
        len_s_1, len_s_2, len_c = len(s_1), len(s_2), len(c)

        input_ids_i = [bos] + s_1 + [eos] + s_2 + [eos] + c + [eos]
        input_ids.append(input_ids_i)
        attention_mask_record.append(1 * [4] + len_s_1 * [1] + 1 * [5] +
                                      len_s_2 * [2] + 1 * [5] +
                                      len_c * [3] + 1 * [5])
        attention_mask.append(1 * [1] + len_s_1 * [1] + 1 * [1] +
                                      len_s_2 * [1] + 1 * [1] +
                                      len_c * [1] + 1 * [1])


    return input_ids, attention_mask_record, attention_mask


def scale_to_range(labels, _min, _max):
    return list(map(lambda x: (x - _min) / (_max - _min), labels))


def get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        condition_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        scale=None,
        condition_only=False,
        sentences_only=False,
        ):

    def preprocess_function(examples):
        result = {'input_ids': [], 'attention_mask_record': [], 'attention_mask': []}
        input_ids, attention_mask_record, attention_mask = tokenize_decoupling(examples[sentence1_key], examples[sentence2_key], examples[condition_key], tokenizer)
        result['input_ids'] = input_ids
        result['attention_mask_record'] = attention_mask_record
        result['attention_mask'] = attention_mask

        result['labels'] = examples[similarity_key]
        if scale is not None:
            _min, _max = scale
            for label in result['labels']:
                if (label < _min or label > _max) and label != -1:
                    raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
            result['labels'] = scale_to_range(result['labels'], _min, _max)
        return result

    return preprocess_function
