# 标准 Python 和 gRPC 相关的依赖
import os
import sys
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
import grpc
import torch
import numpy as np

# ROOT_DIR = ... 和 sys.path.append 用于把项目上的本地包（包括 cosyvoice 以及 Matcha-TTS 等）加入 Python Path，确保本地模块导入时能正常找到。
# from cosyvoice.cli.cosyvoice import CosyVoice2 表明我们要使用 CosyVoice2 这个类（而不是官方示例中的 CosyVoice 类）。
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2

# 设置日志级别为 DEBUG，方便调试时输出更多的信息。
# 设置日志格式，以便更好地查看时间、日志级别、具体日志内容。
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

def __init__(self, args):
        try:
            self.cosyvoice = CosyVoice2(args.model_dir)
            logging.info('CosyVoice2 model loaded successfully')
        except Exception as e:
            logging.error(f'Failed to load CosyVoice2 model: {e}')
            raise RuntimeError('Failed to load CosyVoice2 model')

try:
    self.cosyvoice = CosyVoice2(args.model_dir)
    logging.info('CosyVoice2 model loaded successfully')
except Exception as e:
    logging.error(f'Failed to load CosyVoice2 model: {e}')
    raise RuntimeError('Failed to load CosyVoice2 model')

# （1）从本地文件读取提示音频
local_prompt_path = os.path.join(ROOT_DIR, "prompt_audios", "standard_female.wav")
prompt_speech, sr = torchaudio.load(local_prompt_path)  # 返回形状通常是 [channels, waveform_length]

# （2）如果不是 16k 采样率，可以用 torchaudio 进行重采样
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    prompt_speech = resampler(prompt_speech)

# （3）如果是立体声或多声道，这里可取均值转换为单声道（根据实际需要可选）
# prompt_speech = prompt_speech.mean(dim=0, keepdim=True)  # [1, waveform_length]

# （4）CosyVoice2 预期输入通常是 [batch_size, waveform_length]
# torchaudio.load 返回 [channels, waveform_length]，通常要保证 batch_size 维度在最前面
# 若使用单声道可直接把 [1, waveform_length] 当作 [batch_size, waveform_length]
prompt_speech_16k = prompt_speech

# （5）从请求中获取文本内容
# tts_text: 希望合成的目标文本
# prompt_text: 提示音频对应的文本（可用于辅助模型理解/学习音色）
tts_text = request.zero_shot_request.tts_text
prompt_text = request.zero_shot_request.prompt_text

# （6）调用 CosyVoice2 的零样本推理接口
model_output = self.cosyvoice.inference_zero_shot(
    tts_text,
    prompt_text,
    prompt_speech_16k
)