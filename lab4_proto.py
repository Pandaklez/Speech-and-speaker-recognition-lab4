
# DT2119, Lab 4 End-to-end Speech Recognition
import torchaudio
import torch
from torch.utils.data import DataLoader
import numpy as np
import kenlm
from pyctcdecode import build_ctcdecoder


# Functions to be implemented ----------------------------------


def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    chars = list("' abcdefghijklmnopqrstuvwxyz")
    string_of_chars = ' '.join([chars[n] for n in labels]).upper()
    return string_of_chars

def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    chars = list("' abcdefghijklmnopqrstuvwxyz")
    ints = [chars.index(c) for c in list(text.lower())]
    return ints


def dataProcessing(data, transform=[]):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band. C is 1 for mono-channel audio.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    spectrograms, labels_before_padding, input_lengths, label_lengths = [], [], [], []
    # print(data)
    for waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id in data:
        if utterance == '':
            print("EMPTY UTTERANCE, skipping...")
            continue
        encoded_utterance = strToInt(utterance)
        labels_before_padding.extend(encoded_utterance)
        label_lengths.append(len(encoded_utterance))
        # print("waveform before turining into spectogram: ", waveform)

        if 'augment' in transform:
            # spectrogram masking
            freq_mask_param = 15
            time_mask_param = 35
            spec_aug = torch.nn.Sequential(
                    torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80),
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
                    torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param),
                )
            mspectogram = spec_aug(waveform)
        else:
            # apply transform to waveform
            mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80)  # number of mel filterbanks
            mspectogram = mel_transform(waveform)
        
        mspectogram = mspectogram.squeeze(0).transpose(0, 1)

        spectrograms.append(mspectogram)
        # list of integers Li = Ti/2 where Ti is the number of frames in spectrogram i (before padding)
        # print("mspectogram.size() = ", mspectogram.size())  # == (time, n_mels)
        input_lengths.append(int(mspectogram.size(0)/2))

    # pad spectrograms and labels
    # print("spectrograms size before padding = ", len(spectrograms), spectrograms[0].size())  # 30 torch.Size([1216, 80]) batch_size [T, n_mels]
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    # print("spectrograms.size() after padding B x C x M x T = ", spectrograms.size())   #  B x C x M x T  [30, 1, 80, 1422
    # print("len(labels_before_padding) = ", len(labels_before_padding))
    # labels_before_padding = [torch.tensor(el) for el in labels_before_padding]
    labels_before_padding = torch.tensor(labels_before_padding)
    # labels = torch.nn.utils.rnn.pad_sequence(labels_before_padding, batch_first=True)
    #print("labels.size() B x L = ", labels.size())    # B x L  torch.Size([256, 338])
    #print("labels after padding = ", labels)
    #print("label_lengths before padding: ", label_lengths)
    #print("label_lengths.sum(), len(labels_before_padding): ", sum(label_lengths), len(labels_before_padding))
    #print("len of input_lengths: ", len(input_lengths))
    #print("Check input_length > label_length: ", [a > b for a, b in zip(input_lengths, label_lengths)])
    time = spectrograms.size(3)
    for el in input_lengths:
        if el > time:
            print("input_lengths > time which is BAD: ", el, time)
    return spectrograms, labels_before_padding, input_lengths, label_lengths
    
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output (probabilities) tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    # Extract a sequence containing the most probable character for each time step of the output
    # output = output.cpu().detach().numpy()
    # print("Output to greedy decoder min mean max: ", np.min(output), np.mean(output), np.mean(output))
    print("Outputs preargmax shape: ", output.shape)  # (30, 630, 29)
    # Iterate through each pair of timestamps
    #similarity_results = []
    #for i in range(output.shape[1] - 1):  # -1 to avoid index out of range in the last iteration
    #    is_close = np.isclose(output[:, i, :], output[:, i + 1, :])
    #    similarity_results.append(is_close)

    # Convert the list of results to a numpy array for easier handling
    #similarity_results = np.array(similarity_results)
    # Check overall similarity
    #overall_similarity = np.all(similarity_results, axis=(0, 2))
    #print("Overall similarity between consecutive timestamps: ", overall_similarity)
    decoded = []
    max_prob_indices = np.argmax(output, axis=-1) 
    print("max_prob_indices.shape in greedy decoder (B, T): ", max_prob_indices.shape)  #  (30, 1269)
    # print("max_prob_indices in greedy decoder (B, T): ", max_prob_indices)  # returns indices of the max prob for each character
    print("Check whether (max_prob_indices==blank_label).all(): ", (max_prob_indices==blank_label).all())
    for batch in max_prob_indices:
        # print("batch.shape in greedy decoder iter (T,C): ", batch.shape)  # (1269,)
        # print("batch in greedy decoder iter (T,C): ", batch)  # returns negative loglikelihoods
        decoded.append([])
        # Merge any adjacent identical characters
        prev_ind = None
        for k, ind in enumerate(batch):
            if k == 0 or ind != prev_ind:
                if ind != blank_label:
                    decoded[-1].append(ind)
            prev_ind = ind
    decoded = [intToStr(d) for d in decoded]
    return decoded


def pyctcDecoder(decoder, output, alpha=0.5, beta=1.0, blank_label=28, grid=False):
    decoded = []
    if not grid:
        for batch in output:
            decoded.append(decoder.decode(batch))
    elif grid:
        decoded = decoder.decode(output)
    return decoded

def levenshteinDistance(ref, hyp):
    '''
    Calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    len_ref = len(ref)
    len_hyp = len(hyp)
    distances = np.zeros((len_ref + 1, len_hyp + 1))
    distances[0] = np.arange(len_hyp + 1)
    distances[:, 0] = np.arange(len_ref + 1)
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i-1] == hyp[j-1]:
                distances[i][j] = distances[i-1][j-1]
            else:
                distances[i][j] = min(distances[i-1][j-1], distances[i][j-1], distances[i-1][j]) + 1
    return distances[len_ref][len_hyp]


"""
# reload torch datasets from tar.gz files
train_dataset = torchaudio.datasets.LIBRISPEECH('train-clean-100.tar.gz', url='train-clean-100', download=False)
dev_dataset = torchaudio.datasets.LIBRISPEECH('dev-clean.tar.gz', url='dev-clean', download=False)
test_dataset = torchaudio.datasets.LIBRISPEECH('test-clean.tar.gz', url='test-clean', download=False)

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = LibriSpeechLoader(train_dataset, train=True, transform=['augment'])
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = LibriSpeechLoader(test_dataset, train=False, transform=[])
dev_audio_transform = LibriSpeechLoader(dev_dataset, train=False, transform=[])
"""
