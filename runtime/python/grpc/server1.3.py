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
import torchaudio
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

class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = CosyVoice2(args.model_dir)
            logging.info('CosyVoice2 model loaded successfully')
        except Exception as e:
            logging.error(f'Failed to load CosyVoice2 model: {e}')
            raise RuntimeError('Failed to load CosyVoice2 model')

    def Inference(self, request, context):
        if request.HasField('zero_shot_request'):
            logging.info('Received zero-shot inference request')

            local_prompt_path = os.path.join(ROOT_DIR, "prompt_audios", "Roise.wav")
            prompt_speech, sr = torchaudio.load(local_prompt_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                prompt_speech = resampler(prompt_speech)
            prompt_speech_16k = prompt_speech

            tts_text = request.zero_shot_request.tts_text
            prompt_text = request.zero_shot_request.prompt_text

            model_output = list(self.cosyvoice.inference_zero_shot(
                tts_text,
                prompt_text,
                prompt_speech_16k,
                stream=False
            ))

            if len(model_output) == 0:
                logging.error("Model output is empty!")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Inference output is empty")
                return
            
            logging.info(f"Generated {len(model_output)} speech chunks")
            logging.info(f"First speech tensor shape: {model_output[0]['tts_speech'].shape}")

        elif request.HasField('cross_lingual_request'):
            logging.info('Received cross-lingual inference request')
            prompt_speech_16k = torch.from_numpy(
                np.array(np.frombuffer(request.cross_lingual_request.prompt_audio, dtype=np.int16))
            ).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = list(self.cosyvoice.inference_cross_lingual(
                request.cross_lingual_request.tts_text, prompt_speech_16k
            ))

        elif request.HasField('instruct_request'):
            logging.info('Received instruct inference request')
            prompt_speech_16k = torch.from_numpy(
                np.array(np.frombuffer(request.instruct_request.prompt_audio, dtype=np.int16))
            ).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = list(self.cosyvoice.inference_instruct2(
                request.instruct_request.tts_text,
                request.instruct_request.instruct_text,
                prompt_speech_16k
            ))
        else:
            logging.warning('Invalid request type')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Invalid request type')
            return
        
        logging.info('Sending inference response')
        
        server_audio = torch.cat([i['tts_speech'] for i in model_output], dim=1)
        output_path = os.path.join(ROOT_DIR, "server_generated.wav")
        torchaudio.save(output_path, server_audio, 16000, format="wav")
        logging.info(f"Saved server-generated audio to {output_path}")

        for i in model_output:
            response = cosyvoice_pb2.Response()
            response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            yield response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max_conc', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='iic/CosyVoice2-0.5B', help='local path or modelscope repo id')
    args = parser.parse_args()

    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()
    logging.info("Server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()

if __name__ == '__main__':
    main()
