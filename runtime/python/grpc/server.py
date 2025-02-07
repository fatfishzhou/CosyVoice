# 标准 Python 和 gRPC 相关的依赖
import os
import sys
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
import torch
import grpc
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

# 这里 CosyVoiceServiceImpl 继承了 cosyvoice_pb2_grpc.CosyVoiceServicer，这也是官方示例的思路：实现 gRPC 的 servicer，并在其中自定义实际的推理逻辑。
# 直接调用 CosyVoice2(args.model_dir) 初始化模型对象。如果成功则在日志里打印“model loaded successfully”，失败则报错。
class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            # 初始化模型
            self.cosyvoice = CosyVoice2(args.model_dir,fp16=True)
            logging.info('CosyVoice2 model loaded successfully')

            # ✅ 初始化全局 prompt_audio
            local_prompt_path = os.path.join(ROOT_DIR, "prompt_audios", "Roise.wav")
            prompt_speech = torchaudio.load(local_prompt_path)
            self.global_prompt_audio = prompt_speech

            # ✅ 初始化全局 prompt_text
            self.global_prompt_text = "卡迪夫我还没有听过，反正南方那些城市我都没怎么玩过，我去过最南的地方大概就是伦敦了"

            logging.info('服务器已初始化 prompt_audio 和 prompt_text')

        except Exception as e:
            logging.error(f'Failed to load CosyVoice2 model: {e}')
            raise RuntimeError('Failed to load CosyVoice2 model')

        
    def VoiceChange(self, request, context):
        """处理客户端上传的新 prompt_audio 和 prompt_text"""
        logging.info("🚀 Received new voice change request")

        self.global_prompt_audio = torch.from_numpy(
            np.array(np.frombuffer(request.prompt_audio, dtype=np.int16))
        ).unsqueeze(dim=0).float() / (2**15)
        logging.info("✅ 服务器已更新 prompt_audio")
        
        self.global_prompt_text = request.prompt_text
        logging.info(f"✅ 服务器已更新 prompt_text: {self.global_prompt_text}")

        # 立即返回确认信息
        response = cosyvoice_pb2.Response()
        response.tts_audio = b""  # 这里不需要返回音频
        yield response

# 服务端最重要的地方：处理来自 gRPC 客户端的推理请求，并返回推理结果。
    def Inference(self, request, context):
        # 识别 gRPC 客户端传来的请求类型到底是哪一种
        # （1）从本地文件读取提示音频
        local_prompt_path = os.path.join(ROOT_DIR, "prompt_audios", "Roise.wav")
        prompt_speech = torchaudio.load(local_prompt_path)  # 返回形状通常是 [channels, waveform_length]

        # 模仿目标声音、输出文字、模仿目标声音文字
        prompt_speech_16k = prompt_speech
        tts_text = request.zero_shot_request.tts_text
        prompt_text = "卡迪夫我还没有听过，反正南方那些城市我都没怎么玩过，我去过最南的地方大概就是伦敦了"
        

        if request.HasField('zero_shot_request'):
            logging.info('Received zero-shot inference request')
            model_output = self.cosyvoice.inference_zero_shot(
                tts_text,
                prompt_text=self.global_prompt_text,
                prompt_speech_16k=self.global_prompt_audio
            )
        elif request.HasField('cross_lingual_request'):
            logging.info('Received cross-lingual inference request')
            model_output = self.cosyvoice.inference_cross_lingual(
                request.cross_lingual_request.tts_text, 
                prompt_speech_16k=self.global_prompt_audio
            )
        elif request.HasField('instruct_request'):
            logging.info('Received instruct inference request')
            model_output = self.cosyvoice.inference_instruct2(
                request.instruct_request.tts_text,
                request.instruct_request.instruct_text,
                prompt_speech_16k=self.global_prompt_audio
            )            
        else:
            logging.warning('Invalid request type')
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Invalid request type')
            return
        
        logging.info('Sending inference response')
        # for i in model_output: ... yield response 表明每次从模型的 model_output 里取一段音频，就即时返回给客户端一段 Response。客户端会以流（stream）的形式接收多段数据
        for i in model_output:
            response = cosyvoice_pb2.Response()
            response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            yield response


def main():
    # argparse.ArgumentParser()：用于解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max_conc', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='iic/CosyVoice2-0.5B', help='local path or modelscope repo id')
    # 解析用户输入的命令行参数，并存入 args 变量
    args = parser.parse_args()

    # 创建 gRPC 服务器
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    # 创建 CosyVoiceServiceImpl 实例,它是 gRPC 的 servicer，负责处理客户端的请求
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    # 服务器监听所有可用的网络接口（即允许外部设备访问）
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()
    logging.info("Server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    main()
