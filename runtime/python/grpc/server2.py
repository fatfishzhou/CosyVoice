import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
import grpc
import torch
import numpy as np

# 关闭 Matplotlib 的警告
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# 导入 CosyVoice2
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# 指定音色素材存储目录
PROMPT_AUDIO_DIR = os.path.join(ROOT_DIR, "prompt_audios")

class CosyVoice2ServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=False)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CosyVoice2: {e}")
        logging.info("CosyVoice2 gRPC service initialized")

    def find_prompt_audio(self, prompt_text):
        """
        在服务器端查找对应的参考音频（默认以文本作为 key）。
        """
        # 这里假设 prompt_text 是一个可以映射到文件名的标识
        prompt_audio_path = os.path.join(PROMPT_AUDIO_DIR, f"{prompt_text}.wav")
        if not os.path.exists(prompt_audio_path):
            logging.error(f"Prompt audio not found: {prompt_audio_path}")
            return None
        return load_wav(prompt_audio_path, 16000)

    def Inference(self, request, context):
        if request.HasField('sft_request'):
            logging.info('Received SFT inference request')
            model_output = self.cosyvoice.inference_sft(
                request.sft_request.tts_text,
                request.sft_request.spk_id
            )

        elif request.HasField('zero_shot_request'):
            logging.info('Received zero-shot inference request')
            prompt_speech_16k = self.find_prompt_audio(request.zero_shot_request.prompt_text)
            if prompt_speech_16k is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Prompt audio not found")
                return
            
            model_output = self.cosyvoice.inference_zero_shot(
                request.zero_shot_request.tts_text,
                request.zero_shot_request.prompt_text,
                prompt_speech_16k
            )

        elif request.HasField('cross_lingual_request'):
            logging.info('Received cross-lingual inference request')
            prompt_speech_16k = self.find_prompt_audio(request.cross_lingual_request.prompt_text)
            if prompt_speech_16k is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Prompt audio not found")
                return
            
            model_output = self.cosyvoice.inference_cross_lingual(
                request.cross_lingual_request.tts_text,
                prompt_speech_16k
            )

        else:
            logging.info('Received instruct inference request')
            model_output = self.cosyvoice.inference_instruct(
                request.instruct_request.tts_text,
                request.instruct_request.spk_id,
                request.instruct_request.instruct_text
            )

        logging.info('Sending inference response')
        for i in model_output:
            response = cosyvoice_pb2.Response()
            response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            yield response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max_conc', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc))
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoice2ServiceImpl(args), server)
    server.add_insecure_port(f"0.0.0.0:{args.port}")
    server.start()
    logging.info(f"Server listening on 0.0.0.0:{args.port}")
    server.wait_for_termination()


if __name__ == '__main__':
    main()