# Speech-and-speaker-recognition-lab4
course assignment

deep_speech_2016.ipynb -- main results

### Comparing results with and without language model:

Subjectively, adding language models obviously recognizes better some words, like turnine "ave" into "of", "P R O S E S S" into "proves" (however, the target was "proves this") etc. So it's good for shaping words that almost right into grammatical form wihtout misspellings, but sometimes it predicts it wrong. For example, it's bad for recognizing unique names.


After training for 13 epochs:

```
Average loss: 0.6938

Test set with Greedy Decoder: Average CER: 0.152721 Average WER: 0.5541

Test set with pyCTC Decoder: Average CER: 0.169338 Average WER: 0.3616
```

**WER is much lower for pyCTCDecoder. CER is comarable for both decoders, but a little better for GreedyDecoder.**

### Grid search

Best alpha and beta after grid search:

```
Best alpha: 0.525, Best beta: 1.0, Best WER: 0.3805722415694304
```

Scores without language model:

```
Average test loss: 0.6938

Test set with Greedy Decoder: Average CER: 0.152721 Average WER: 0.5541
```

Non optimized decoder = Average CER: 0.169338 Average WER: 0.3616

Scores with language model on tuned alpha beta for pyCTC decoder:

```
Test set with optimized pyCTC Decoder: 

Average test loss: 0.6938

Average CER: 0.169864 Average WER: 0.3609
```

**Opitimized pyCTC decoder is only a tiny bit better on WER than a non-optimized one, because after grid search the optimized alpha and beta turned out to be very close to default ones.**

### Recognize mode

**wavfile: 19.wav**

```
text: ['  O I Y   I   O W M Y S   T L O O N     W T   S O   M O N   G O O N   O   O N I N   O E   T R O E N G W H E     M O R D   R U   O   P R O O       O    ']

pyctcdecoded_preds:  ['ii ones to got so on grown or one or throne marred roof poor to h']

target: i also understand that similar branch organizattions have made their appearance in europe
```

**wavfile: 20.wav**

```
text: ['  E T   G O O   M A   H A N T O   C O O M   M E R O N   C O R B E   W H O D O D   N E   C O   O D   G O   A G E H D   T   C R U L   T R E O   H O N D   H O L R Y W S O N   O   N     F O  ']

pyctcdecoded_preds: ['et gone mare anta come care who did not core or grunge tag proud to und holeson on from i']

target: a combination of canadian capital quickly organized and petioned for the same privileges
```

It performs much worse on recordings of my voice. I think it's because the quality of recording and conditions are completely different.
