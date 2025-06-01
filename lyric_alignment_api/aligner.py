import torch
from lyric_alignment_api.model_handling import Wav2Vec2ForCTC
from transformers import AutoTokenizer, AutoFeatureExtractor
from lyric_alignment_api import utils

model_path = 'nguyenvulebinh/lyric-alignment'
use_gpu = torch.cuda.is_available()

# Model, tokenizer, extractor, vocab
model = None
tokenizer = None
feature_extractor = None
vocab = None

def load_model():
    global model, tokenizer, feature_extractor, vocab
    model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
    if use_gpu:
        model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer.get_vocab()))]

def do_asr(waveform):
    input_values = feature_extractor.pad([
        {"input_values": feature_extractor(item, sampling_rate=16000)["input_values"][0]} for item in waveform
    ], return_tensors='pt')["input_values"]

    if use_gpu:
        input_values = input_values.cuda()

    out_values = model(input_values=input_values)
    logits = out_values.logits[0]
    emissions = torch.log_softmax(logits, dim=-1)
    emission = emissions.cpu().detach()
    emission[emission < -20] = -20

    # Adjust special tokens
    pipe_token = tokenizer.convert_tokens_to_ids('|')
    pad_token = tokenizer.convert_tokens_to_ids('<pad>')
    emission[:, pipe_token] = torch.max(emission[:, pipe_token], emission[:, pad_token])
    emission[:, pad_token] = -20
    return emission

def do_force_align(waveform, emission, word_list, sample_rate=16000, base_stamp=0):
    transcript = '|'.join(word_list)
    dictionary = {c: i for i, c in enumerate(vocab)}
    tokens = [dictionary.get(c, 0) for c in transcript]
    trellis = utils.get_trellis(emission, tokens, blank_id=tokenizer.convert_tokens_to_ids('|'))
    path = utils.backtrack(trellis, emission, tokens)
    segments = utils.merge_repeats(path, transcript)
    word_segments = utils.merge_words(segments)
    word_segments = utils.add_pad(word_segments, emission)
    ratio = waveform.size(1) / (trellis.size(0) - 1)

    result = []
    for word in word_segments:
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        result.append({
            'd': word.label,
            's': int(x0 / sample_rate * 1000) + base_stamp,
            'e': int(x1 / sample_rate * 1000) + base_stamp
        })
    assert [item['d'] for item in result] == word_list
    return result

def align_lyrics(wav, lyric_alignment_json):
    seg_words = [[word['d'] for word in seg['l']] for seg in lyric_alignment_json]
    words = [y for x in seg_words for y in x]
    words_norm = [utils.norm_word(word) for word in words]
    single_words_list = [word.split() for word in words_norm]
    single_words = [y for x in single_words_list for y in x]

    emission = do_asr(wav)
    word_piece = do_force_align(wav, emission, single_words)

    # Ghép ngược lại vào cấu trúc lyrics
    words_align_result = []
    word_piece_idx = 0
    for idx in range(len(words)):
        len_single_words = len(single_words_list[idx])
        list_piece_align = word_piece[word_piece_idx: word_piece_idx + len_single_words]
        word_piece_idx += len_single_words
        words_align_result.append(list_piece_align)

    seg_words_align_result = []
    word_idx = 0
    for idx in range(len(seg_words)):
        len_seg = len(seg_words[idx])
        list_word_align = words_align_result[word_idx: word_idx + len_seg]
        word_idx += len_seg
        seg_words_align_result.append(list_word_align)

    for list_word_align, segment_info in zip(seg_words_align_result, lyric_alignment_json):
        for idx, (word_align, word_raw) in enumerate(zip(list_word_align, segment_info['l'])):
            if len(word_align) > 0:
                word_raw['s'] = word_align[0]['s']
                word_raw['e'] = word_align[-1]['e']
            elif idx > 0:
                word_raw['s'] = segment_info['l'][idx - 1]['e']
                word_raw['e'] = segment_info['l'][idx - 1]['e']
        if len(segment_info['l']) > 0:
            segment_info['s'] = segment_info['l'][0]['s']
            segment_info['e'] = segment_info['l'][-1]['e']

    return lyric_alignment_json
