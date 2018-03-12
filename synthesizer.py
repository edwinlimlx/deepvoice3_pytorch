# coding: utf-8
import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects

"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
import sys
import os
from os.path import dirname, join, basename, splitext

import audio

import torch
from torch.autograd import Variable
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams

from tqdm import tqdm

use_cuda = torch.cuda.is_available()

class Synthesizer:
  def tts(self, model, text, p=0.0, speaker_id=None, fast=False, _frontend=None):
    if use_cuda:
      model = model.cuda()
    model.eval()
    if fast:
      model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    speaker_ids = None if speaker_id is None else Variable(torch.LongTensor([speaker_id]))
    if use_cuda:
      sequence = sequence.cuda()
      text_positions = text_positions.cuda()
      speaker_ids = None if speaker_ids is None else speaker_ids.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments, _ = model(sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel

  def load(self, checkpoint_path, hparams):
    self.speaker_id = None
    if self.speaker_id is not None:
      self.speaker_id = int(self.speaker_id)

    print(hparams)
    max_decoder_steps = 200

    # Override hyper parameters
    hparams.parse("use_preset=True,builder=deepvoice3")
    assert hparams.name == "deepvoice3"

    # Presets
    if hparams.use_preset:
      preset = hparams.presets[hparams.builder]
      import json
      hparams.parse_json(json.dumps(preset))
      print("Override hyper parameters with preset \"{}\": {}".format(hparams.builder, json.dumps(preset, indent=4)))

    self._frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = self._frontend
    from train import build_model

    # Model
    self.model = build_model()

    # Load checkpoints separately
    checkpoint = torch.load(checkpoint_path)
    self.model.load_state_dict(checkpoint["state_dict"])
    self.checkpoint_name = splitext(basename(checkpoint_path))[0]

    self.model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    print('Loading checkpoint: %s' % checkpoint_path)


  def synthesize(self, text):
    print("text: " + text + ", speaker_id: " + str(self.speaker_id), self.checkpoint_name)

    waveform, alignment, _, _ = self.tts(self.model, text, p=0.0, speaker_id=self.speaker_id, fast=True, _frontend=self._frontend)
    dst_wav_path = join("./output/", "{}.wav".format(self.checkpoint_name))
    # dst_alignment_path = join(dst_dir, "{}_{}{}_alignment.png".format(idx, checkpoint_name))
    # plot_alignment(alignment.T, dst_alignment_path,info="{}, {}".format(hparams.builder, basename(self.checkpoint_path)))

    out = io.BytesIO()
    audio.save_wav(waveform, out)
    return out.getvalue()
