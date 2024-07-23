# DT2119, Lab 4 End-to-end Speech Recognition
import os
import torch
from torch import nn
import torchaudio
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import argparse
from lab4_proto import *
import kenlm
from pyctcdecode import build_ctcdecoder
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

'''
HYPERPARAMETERS
'''
hparams = {
	"n_cnn_layers": 3,
	"n_rnn_layers": 5,
	"rnn_dim": 512,
	"n_class": 29,
	"n_feats": 80,
	"stride": 2,
	"dropout": 0.1,
	"learning_rate": 5e-4,
	"batch_size": 30, 
	"epochs": 10
}


'''
MODEL DEFINITION
'''
class CNNLayerNorm(nn.Module):
	"""Layer normalization built for cnns input"""
	def __init__(self, n_feats):
		super(CNNLayerNorm, self).__init__()
		self.layer_norm = nn.LayerNorm(n_feats)

	def forward(self, x):
		# x (batch, channel, feature, time)
		x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
		x = self.layer_norm(x)
		return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 

class ResidualCNN(nn.Module):
	"""Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
		except with layer norm instead of batch norm
	"""
	def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
		super(ResidualCNN, self).__init__()

		self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
		self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.layer_norm1 = CNNLayerNorm(n_feats)
		self.layer_norm2 = CNNLayerNorm(n_feats)

	def forward(self, x):
		residual = x  # (batch, channel, feature, time)
		x = self.layer_norm1(x)
		x = F.gelu(x)
		x = self.dropout1(x)
		x = self.cnn1(x)
		x = self.layer_norm2(x)
		x = F.gelu(x)
		x = self.dropout2(x)
		x = self.cnn2(x)
		x += residual
		return x # (batch, channel, feature, time)
		
class BidirectionalGRU(nn.Module):

	def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
		super(BidirectionalGRU, self).__init__()

		self.BiGRU = nn.GRU(
			input_size=rnn_dim, hidden_size=hidden_size,
			num_layers=1, batch_first=batch_first, bidirectional=True)
		self.layer_norm = nn.LayerNorm(rnn_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		#print('bi-gru, in:',x.shape)
		x = self.layer_norm(x)
		x = F.gelu(x)
		x, _ = self.BiGRU(x)
		x = self.dropout(x)
		#print('bi-gru, out:',x.shape)
		return x

class SpeechRecognitionModel(nn.Module):
	"""Speech Recognition Model Inspired by DeepSpeech 2"""

	def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
		super(SpeechRecognitionModel, self).__init__()
		n_feats = n_feats//stride
		self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

		# n residual cnn layers with filter size of 32
		self.rescnn_layers = nn.Sequential(*[
			ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
			for _ in range(n_cnn_layers)
		])
		self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
		self.birnn_layers = nn.Sequential(*[
			BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
							 hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
			for i in range(n_rnn_layers)
		])
		self.classifier = nn.Sequential(
			nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(rnn_dim, n_class),
			nn.LogSoftmax(dim=2)
		)

	def forward(self, x):
		x = self.cnn(x)
		x = self.rescnn_layers(x)
		sizes = x.size()
		x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
		x = x.transpose(1, 2)  # (batch, time, feature)
		x = self.fully_connected(x)
		x = self.birnn_layers(x)
		x = self.classifier(x)
		return x

'''
ACCURACY MEASURES
'''
def wer(reference, hypothesis, ignore_case=False, delimiter=' ', greedy=False):
	if ignore_case == True:
		reference = reference.lower()
		hypothesis = hypothesis.lower()

	ref_words = reference.split('  ')
	ref_words = [word.replace(' ', '') for word in ref_words]

	if greedy:
		hyp_words = hypothesis.split('  ')
		hyp_words = [word.replace(' ', '') for word in hyp_words]
	else:
		hyp_words = hypothesis.split(delimiter)
	edit_distance = levenshteinDistance(ref_words, hyp_words)
	ref_len = len(ref_words)

	if ref_len > 0:
		wer = float(edit_distance) / ref_len
	else:
		raise ValueError("empty reference string")  
	return wer

def cer(reference, hypothesis, ignore_case=False, remove_space=False):
	if ignore_case == True:
		reference = reference.lower()
		hypothesis = hypothesis.lower()

	join_char = ' '
	if remove_space == True:
		join_char = ''

	reference = join_char.join(filter(None, reference.split(' ')))
	hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

	edit_distance = levenshteinDistance(reference, hypothesis)
	ref_len = len(reference)
	if ref_len > 0:
		cer = float(edit_distance) / ref_len
	else:
		raise ValueError("empty reference string")
	return cer

'''
TRAINING AND TESTING
'''

def train(model, device, train_loader, criterion, optimizer, epoch):
	model.train()
	data_len = len(train_loader.dataset)
	batch_idx = 0
	for _data in train_loader:
		print('batch_idx:', batch_idx)
		# print('_data:', _data)
		spectrograms, labels, input_lengths, label_lengths = _data 
		spectrograms, labels = spectrograms.to(device), labels.to(device)

		optimizer.zero_grad()
		# model output is (batch, time, n_class)
		# print('spectrograms on train mean on time d:', spectrograms.mean())
		# print('spectrograms on train min:', spectrograms.min())
		# print('spectrograms on train max:', spectrograms.max())
		output = model(spectrograms)
		# transpose to (time, batch, n_class) in loss function == torch.Size([668, 30, 29])
		# print('output.transpose(0, 1).shape, transpose to (time, batch, n_class) in loss function: ', output.transpose(0, 1).shape)
		# print('Model output on train mean:', output.transpose(0, 1).mean())
		#print('Model output on train min argmin(dim=0) on time d:', output.transpose(0, 1).argmin(dim=0))
		#print('Model output on train max argmax(dim=0) on time d:', output.transpose(0, 1).argmax(dim=0))
		# print('labels.shape in train: ', labels.shape) # torch.Size([30, 277]) (batch, )
		#print('labels in train: ', labels)
		#print(input_lengths, label_lengths)
		print('input_lengths.shape in train: ', torch.tensor(input_lengths).shape)
		print('label_lengths.shape in train: ', torch.tensor(label_lengths).shape)
		# transpose to (time, batch, n_class) in loss function  (T,B,C)
		output = output.transpose(0, 1)
		print('Model output on train size should be (time, batch, n_class): ', output.size())  # [_, 30, 29]
		loss = criterion(output, labels, torch.tensor(input_lengths), torch.tensor(label_lengths))
		# print('CTC loss on batch step:', loss)
		loss.backward()
		optimizer.step()
		
		if (batch_idx % 200 == 0 and batch_idx != 0) or batch_idx == data_len:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(spectrograms), data_len,
				100. * batch_idx / len(train_loader), loss.item()))
		#if batch_idx % 500 == 0 and batch_idx != 0:
			# Validation
		#	validation(model, device, val_loader, criterion, epoch)
		#	model.train()
		
		batch_idx += 1

def validation(model, device, val_loader, criterion, epoch):
	print('\nValidation run…')
	model.eval()
	val_loss = 0
	decoded_targets, outputs = [], []
	with torch.no_grad():
		for I, _data in tqdm(enumerate(val_loader), total=len(val_loader)):
			spectrograms, labels, input_lengths, label_lengths = _data 
			spectrograms, labels = spectrograms.to(device), labels.to(device)

			output = model(spectrograms)
			output = output.cpu().detach().numpy()

			for output, length in zip(output, input_lengths):
				outputs.append(output[:length])

			last_ind = 0
			for i in range(len(label_lengths)):
				labels_in_range = labels[last_ind:last_ind+label_lengths[i]].tolist()
				last_ind = last_ind + label_lengths[i]
				decoded_targets.append(intToStr(labels_in_range))
	print('Validation average loss: {:.4f}\n'.format(val_loss))
	return decoded_targets, outputs

def test(model, device, test_loader, criterion, epoch, lm=False):
	print('\nEvaluating…')
	model.eval()
	test_loss = 0
	test_cer, test_wer = [], []
	pyctctest_cer, pyctctest_wer = [], []
	labels = list("' abcdefghijklmnopqrstuvwxyz")
	decoder = build_ctcdecoder(
				labels=labels,
				kenlm_model_path="wiki-interpolate.3gram.arpa",  # "/nfs/deepspeech/home/annkle/lab4/lab4/wiki-interpolate.3gram.arpa",
				alpha=0.525,  # the best after grid search
				beta=1.0
			)
	with torch.no_grad():
		for I, _data in tqdm(enumerate(test_loader), total=len(test_loader)):
			spectrograms, labels, input_lengths, label_lengths = _data 
			spectrograms, labels = spectrograms.to(device), labels.to(device)

			# model output is (batch, time, n_class)
			output = model(spectrograms)  
			# transpose to (time, batch, n_class) in loss function  (T,B,C)
			output = output.transpose(0, 1)
			loss = criterion(output, labels, input_lengths, label_lengths)
			test_loss += loss.item() / len(test_loader)

			# get target text
			decoded_targets = []
			#for i in range(len(labels)):
			#	print("labels[i]: ", labels[i])
			#	decoded_targets.append(intToStr(labels[i][:label_lengths[i]].tolist()))
			last_ind = 0
			for i in range(len(label_lengths)):
				labels_in_range = labels[last_ind:last_ind+label_lengths[i]].tolist()
				last_ind = last_ind + label_lengths[i]
				# print("labels_in_range in test: ", labels_in_range)
				decoded_targets.append(intToStr(labels_in_range))
			# print("decoded_targets:\n", decoded_targets)

			if not lm:
				# get predicted text
				decoded_preds = greedyDecoder(output.transpose(1, 0).cpu().detach().numpy(), blank_label=28)
				print("Greedy decoded batch:\n", decoded_preds)
				# print("len(decoded_preds):", len(decoded_preds))
				# calculate accuracy for Greedy Decoder
				for j in range(len(decoded_preds)):
					test_cer.append(cer(decoded_targets[j], decoded_preds[j], ignore_case=True))
					test_wer.append(wer(decoded_targets[j], decoded_preds[j], ignore_case=True, greedy=True))

			pyctcdecoded_preds = pyctcDecoder(decoder, output.transpose(1, 0).cpu().detach().numpy(), blank_label=28)
			# print("pyCTC decoded batch:\n", pyctcdecoded_preds)
			# print("len(decoded_targets), len(input_lengths):", len(decoded_targets), len(input_lengths))
				
			# calculate accuracy for pyCTC Decoder
			for j in range(len(pyctcdecoded_preds)):
				pyctctest_cer.append(cer(decoded_targets[j], pyctcdecoded_preds[j], ignore_case=True, remove_space=True))
				pyctctest_wer.append(wer(decoded_targets[j], pyctcdecoded_preds[j], ignore_case=True))

	if not lm:
		avg_cer = sum(test_cer)/len(test_cer)
		avg_wer = sum(test_wer)/len(test_wer)
		print('Test set with Greedy Decoder: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
	
	avg_pyctc_cer = sum(pyctctest_cer)/len(pyctctest_cer)
	avg_pyctc_wer = sum(pyctctest_wer)/len(pyctctest_wer)
	print('Test set with pyCTC Decoder: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_pyctc_cer, avg_pyctc_wer))

def grid_search(alphas, betas):
	# find the best alpha and beta for lowest WER
	model.eval()
	best_alpha, best_beta, best_wer = 0, 0, 1
	wers_per_batch = defaultdict(list)
	for alpha in alphas:
		print("alpha step: ", alpha)
		for beta in betas:
			print("beta step: ", beta)
			labels = list("' abcdefghijklmnopqrstuvwxyz")
			decoder = build_ctcdecoder(
				labels=labels,
				kenlm_model_path="wiki-interpolate.3gram.arpa",  # "/nfs/deepspeech/home/annkle/lab4/lab4/wiki-interpolate.3gram.arpa",
				alpha=alpha,
				beta=beta
			)
			# read outputs.npz
			outputs = np.load('val_output.npz')
			# read val_decoded_tartgets.txt with Path
			val_decoded_targets = Path("val_decoded_targets.txt").read_text().split("\n")
			
			for i, output in enumerate(outputs.values()):
				# print(output.shape, len(val_decoded_targets[i:]))
				pyctcdecoded_preds = pyctcDecoder(decoder, output, alpha=alpha, beta=beta, blank_label=28, grid=True)
				wers_per_batch[(alpha, beta)].append(wer(val_decoded_targets[i], pyctcdecoded_preds, ignore_case=True))

				if i < 100:
					print(val_decoded_targets[i])
					print(pyctcdecoded_preds)
				# print("pyCTC decoded batch:\n", pyctcdecoded_preds)
				# calculate accuracy for pyCTC Decoder
				
				#for j in range(len(pyctcdecoded_preds)):
				#	if j < 100:
			    #		print(decoded_targets[j])
				#		print(pyctcdecoded_preds[j])
			#		wers_per_batch[(alpha, beta)].append(wer(val_decoded_targets[j], pyctcdecoded_preds[j], ignore_case=True))
	
	print("wers_per_batch:\n\n ", wers_per_batch)

	for grid in wers_per_batch:
		print("Start grid search...\n")
		avg_wer = sum(wers_per_batch[grid])/len(wers_per_batch[grid])
		if avg_wer < best_wer:
			best_alpha = grid[0]
			best_beta = grid[1]
			best_wer = avg_wer
	
	return best_alpha, best_beta, best_wer
			

'''
MAIN PROGRAM
'''
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--mode', type=str, help='train, test or recognize')
	argparser.add_argument('--model', type=str, help='model to load', default='')
	argparser.add_argument('--wavfiles', nargs='*',help='wavfiles to recognize')

	args = argparser.parse_args()

	use_cuda = torch.cuda.is_available()
	#use_mps = torch.backends.mps.is_available()
	torch.manual_seed(7)
	device = torch.device("cuda" if use_cuda else "cpu")

	#print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
	# print(f"Is MPS available? {torch.backends.mps.is_available()}")
	# Set the device      
	# device = "mps" if torch.backends.mps.is_available() else "cpu"
	device = "cpu"
	print(f"Using device: {device}")
	path_cuda = '/nfs/deepspeech/home/annkle/lab4/lab4/'
	train_dataset = torchaudio.datasets.LIBRISPEECH(".", url='train-clean-100', download=False)
	val_dataset = torchaudio.datasets.LIBRISPEECH(".", url='dev-clean', download=False)
	test_dataset = torchaudio.datasets.LIBRISPEECH(".", url='test-clean', download=False)

	# print("len(val_dataset), len(test_dataset): ", len(val_dataset), len(test_dataset))  # 2703 2620
	train_audio_transform = ['augment']
	test_audio_transform = []
	kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
	train_loader = data.DataLoader(dataset=train_dataset,
					batch_size=hparams['batch_size'],
					shuffle=True,
					collate_fn=lambda x: dataProcessing(x, train_audio_transform),
					**kwargs)

	val_loader = data.DataLoader(dataset=val_dataset,
					batch_size=hparams['batch_size'],
					shuffle=False,  # True
					collate_fn=lambda x: dataProcessing(x, test_audio_transform),
					**kwargs)

	test_loader = data.DataLoader(dataset=test_dataset,
					batch_size=hparams['batch_size'],
					shuffle=False,
					collate_fn=lambda x: dataProcessing(x, test_audio_transform),
					**kwargs)

	model = SpeechRecognitionModel(
		hparams['n_cnn_layers'], 
		hparams['n_rnn_layers'], 
		hparams['rnn_dim'],
		hparams['n_class'], 
		hparams['n_feats'], 
		hparams['stride'], 
		hparams['dropout']
		).to(device)

	print(model)
	print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

	optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
	criterion = nn.CTCLoss(blank=28, zero_infinity=False).to(device)
	
	print(args.mode)

	if args.mode == 'finetune':
		last_epoch = 11
		checkpoint = torch.load(path_cuda + 'checkpoints/epoch-{}.pt'.format(last_epoch))  # replace `epoch` with the actual epoch number
		# Load the state_dict into the model
		model.load_state_dict(checkpoint)
		for epoch in range(last_epoch + 1, last_epoch + hparams['epochs']):
			train(model, device, train_loader, criterion, optimizer, epoch)
			test(model, device, val_loader, criterion, epoch)
			torch.save(model.state_dict(), '/nfs/deepspeech/home/annkle/lab4/lab4/checkpoints/epoch-{}.pt'.format(epoch))

	if args.model != '':
		model.load_state_dict(torch.load(args.model, map_location=device))

	if args.mode == 'train':
		for epoch in range(hparams['epochs']):
			train(model, device, train_loader, criterion, optimizer, epoch)
			test(model, device, val_loader, criterion, epoch)
			torch.save(model.state_dict(), '/nfs/deepspeech/home/annkle/lab4/lab4/checkpoints/epoch-{}.pt'.format(epoch))

	elif args.mode == 'test':
		test(model, device, test_loader, criterion, -1)

	elif args.mode == 'testlm':
		test(model, device, test_loader, criterion, -1, lm=True)

	elif args.mode == 'recognize':
		for wavfile in args.wavfiles:
			waveform, sample_rate = torchaudio.load(wavfile, normalize=True)
			print('waveform:', waveform.shape)
			mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80)  # number of mel filterbanks
			spectogram = mel_transform(waveform)
			# mspectogram = mspectogram.squeeze(0).transpose(0, 1)
			# spectrogram = test_audio_transform(waveform)
			input = torch.unsqueeze(spectogram, dim=0).to(device)
			output = model(input)
			text = greedyDecoder(output.detach().numpy())

			labels = list("' abcdefghijklmnopqrstuvwxyz")
			decoder = build_ctcdecoder(
				labels=labels,
				kenlm_model_path="wiki-interpolate.3gram.arpa",  # "/nfs/deepspeech/home/annkle/lab4/lab4/wiki-interpolate.3gram.arpa",
				alpha=0.525,  # the best after grid search
				beta=1.0
			)
			pyctcdecoded_preds = pyctcDecoder(decoder, output.cpu().detach().numpy(), blank_label=28)
			print('wavfile:', wavfile)
			print('text:', text)
			print('pyctcdecoded_preds:', pyctcdecoded_preds)

	elif args.mode == 'save_val':
		decoded_targets, outputs = validation(model, device, val_loader, criterion, -1)
		np.savez('val_output.npz', *outputs)
		Path("val_decoded_targets.txt").write_text("\n".join(decoded_targets))

	elif args.mode == 'tunedecoder':
		# grid search for alpha and beta
		alphas = np.linspace(0.05, 1.0, 5)
		betas = np.linspace(0.05, 1.0, 5)
		best_alpha, best_beta, best_wer = grid_search(alphas, betas)
		print("Best alpha: {}, Best beta: {}, Best WER: {}".format(best_alpha, best_beta, best_wer))
