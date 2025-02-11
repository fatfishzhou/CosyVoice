# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


# CosyVoice2 继承自一个基类 CosyVoice（你也可以把它看作是第一代 cosyvoice 的基础）
# 它在原有功能的基础上，扩展了对指令式合成、零样本合成等功能的支持，并结合了更先进的语言模型（LLM）进行推理
class CosyVoice2(CosyVoice):
    # 这是构造函数，负责初始化 CosyVoice2 模型所需的所有组件，包括前端、LLM、声码器等
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        # 判断 model_dir 路径里是否包含字符串 "-Instruct"，如果包含，就设置 self.instruct = True 表示该模型支持/包含“指令式”能力。否则为 False
        self.instruct = True if '-Instruct' in model_dir else False
        # 记录模型目录路径到 self.model_dir
        self.model_dir = model_dir
        # 记录是否使用半精度推理（fp16）到 self.fp16
        self.fp16 = fp16
        # 如果本地不存在 model_dir 指向的目录，调用 snapshot_download(model_dir) 从远程仓库（如 ModelScope / HuggingFace Hub 等）下载对应的模型快照到本地，然后更新 model_dir
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        # 打开 cosyvoice.yaml 配置文件，该文件包含了模型所需的各种超参数（包括前端、后端、采样率、特殊符号、LLM 配置等）
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            # 使用 load_hyperpyyaml 将其解析为一个 Python 字典 configs。同时传入一个 overrides 参数，用于覆盖其中某些字段（如 qwen_pretrain_path），指向 model_dir/CosyVoice-BlankEN，说明可能该 LLM 是基于一个 QWen 预训练模型等
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        # 调用 get_model_type(configs)，确保返回的模型类型是 CosyVoice2Model，如果不是，就抛出一个错误并提示“不要用某某模型路径来初始化 CosyVoice2”
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        # 初始化一个“前端”对象 CosyVoiceFrontEnd，传入配置里的一些关键字段：
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],                         # 文本分词或音素提取相关设置
                                          configs['feat_extractor'],                        # 特征提取器
                                          '{}/campplus.onnx'.format(model_dir),             # ONNX的前缀模型获处理模块
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),  # ONNX的语音特征转token模块
                                          '{}/spk2info.pt'.format(model_dir),               # 说话人信息文件
                                          configs['allowed_special'])                       # 模型支持的特殊标记
        # 从配置中获取采样率，并存到 self.sample_rate，通常 CosyVoice2 使用 16kHz
        self.sample_rate = configs['sample_rate']
        # 如果当前系统没有可用的 GPU（torch.cuda.is_available() == False），但又指定了 load_jit=True / load_trt=True / fp16=True 这三个 GPU 相关的加速/半精度选项，就把它们都改成 False，并发出一个警告日志
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        # 创建了一个具体的 CosyVoice2 模型实例（大语言模型，“流式”声码器或后端声学模型相关的配置，调制/变换器等后续声学处理模块）
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        # 分别加载 LLM、flow、hift 的权重文件（这是关键一步，将磁盘上存储的 .pt 文件加载到 CosyVoice2Model 内存中）
        self.model.load('{}/llm.pt'.format(model_dir),      # 大语言模型的 PyTorch 权重
                        '{}/flow.pt'.format(model_dir),     # flow 结构的解码器/声码器权重
                        '{}/hift.pt'.format(model_dir))     # 额外的调制模块权重
        # 如果 load_jit=True，则调用 self.model.load_jit(...) 去加载一个经过 JIT（TorchScript）压缩或编译好的模型文件，即 flow.encoder.fp16.zip 或 flow.encoder.fp32.zip
        # 这可以替换或加速某些模块（如 encoder）在推理时的运行速度，尤其在 GPU 环境中可起到优化作用
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        # 如果 load_trt=True，调用 self.model.load_trt(...) 来加载 TensorRT 推理引擎（.plan 文件）
        # 这里常见用法是先由 ONNX 转成 .plan，再用 TensorRT 做进一步优化。此处指定了 .mygpu.plan 文件和一个 .onnx 文件
        # 这会替换或加速解码器（flow.decoder.estimator）的推理流程
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        # 删掉临时的 configs，释放内存，也避免将配置信息保存在类属性里。（有时候是为了防止序列化或内存占用过多。）
        del configs

    # 这个方法是 CosyVoice1 里常见的“指令式合成”接口；在 CosyVoice2 中不再使用同名方法，而是提供了 inference_instruct2。
    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    # 这个方法是 CosyVoice2 中的“指令式”推理接口，允许你传入一个额外的 “指令文本”（instruct_text）来指导模型如何生成语音，例如让模型在指定语气、口音、情感下朗读 tts_text。
    # tts_text：要合成的核心文本内容
    # instruct_text：用来指示模型说话的口音、风格、语气、甚至是情感等示例
    # prompt_speech_16k：提示音频，可以是一个音色样本，让模型模仿其中的说话者音色
    # stream=False：是否让模型在生成过程中“流式”地逐段返回音频。设置为 True 时，模型可能一边合成一边 yield 输出；为 False 时则一次性返回
    # speed=1.0：调节生成语音的语速
    # text_frontend=True：是否使用文本前端处理（在上面初始化的 CosyVoiceFrontEnd 里可能包括分词、音素标注等）
    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        # 确保当前 self.model 真的是 CosyVoice2Model 实例，若不是就报错
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        # 调用 self.frontend.text_normalize(...) 对 tts_text 做文本正则化、分句、音素转换等操作
        # 参数 split=True 表示可能把整段文本拆分成若干子句，能让后续合成更灵活、可逐段输出
        # tqdm(...) 给这个循环包了一层进度条，方便查看处理进度
        # 这里的循环 for i in ...: 表示会一条条地取出分好的文本片段（即 i）
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            # 对每个分好的文本片段 i，再进行一次针对“指令模式”的前端处理
            # 得到的 model_input 是一个 Python 字典或类似结构，包含了 TTS 模型所需的所有张量或参数
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            # 调用 self.model.tts(**model_input, stream=stream, speed=speed)，这是 CosyVoice2Model 提供的核心合成函数
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
