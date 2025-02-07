import grpc
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import sounddevice as sd
import numpy as np
import wave
import sys
import time

# 服务器地址（修改为实际 IP）
SERVER_ADDRESS = "20.0.137.8:50000"

# 采样率
SAMPLE_RATE = 24000

def change_voice(voice_place, prompt_text):
    voice_place = input("请输入新的人声路径: ").strip
    prompt_text = input("请输入新人声对应文字用于模型训练矫正").strip

def play_audio_stream(response):
    """接收服务器返回的音频流，并实时播放"""
    audio_buffer = bytearray()
    
    for res in response:  # ✅ 迭代 gRPC 流
        audio_buffer.extend(res.tts_audio)  # 追加音频数据
    
    if len(audio_buffer) > 0:
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
        sd.play(audio_array, samplerate=SAMPLE_RATE)
        sd.wait()

def send_request(mode):
    """发送请求到服务器，并播放和保存返回的 TTS 音频"""
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

    while True:
        text = input("\n请输入要转换的文本 (输入 'exit' 退出): ").strip()
        if text.lower() == "exit":
            break
        

        if mode == "zero_shot":
            request = cosyvoice_pb2.Request(
                zero_shot_request=cosyvoice_pb2.zeroshotRequest(tts_text=text)
            )
        elif mode == "cross_lingual":
            request = cosyvoice_pb2.Request(
                cross_lingual_request=cosyvoice_pb2.crosslingualRequest(tts_text=text)
            )
        elif mode == "instruct":
            instruct_text = input("请输入指令文本 (e.g., '用四川话说这句话'): ").strip()
            request = cosyvoice_pb2.Request(
                instruct_request=cosyvoice_pb2.instructRequest(tts_text=text,instruct_text=instruct_text)
            )
        elif mode == "voice_change":
            prompt_text = input("请输入新人声对应文字用于模型训练矫正: ").strip()
            # 让用户手动输入 `.wav` 文件
            prompt_wav_path = input("请输入要上传的参考音频文件 (同目录下的 .wav 文件名): ").strip()
            # 检查文件是否存在
            if not os.path.exists(prompt_wav_path):
                print(f"❌ 文件 {prompt_wav_path} 不存在，请检查路径！")
                return
            # 读取 .wav 文件
            prompt_speech = load_wav(prompt_wav_path, 16000)

            request = cosyvoice_pb2.Request(
                voice_change_request=cosyvoice_pb2.voicechangeRequest(
                    prompt_text=prompt_text,
                    prompt_audio=(prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
                )
            )
            print("\n🚀 正在上传新的 prompt_text 和 prompt_audio...")
            response = stub.VoiceChange(request)
            for _ in response:
                print("✅ 服务器已成功更新新的人声！")

        else:
            print("模式错误！")
            return

        print("\n🎙️ 服务器处理中...")
        
        # 播放并保存音频
        response = stub.Inference(request)

        play_audio_stream(response)  # ✅ 这样才是正确的

        

if __name__ == "__main__":
    print("\n请选择模式: ")
    print("1 Zero-shot (用文本匹配音色)")
    print("2 Cross-lingual (跨语言)")
    print("3 Instruct (指定风格/口音)")
    print("4 VoiceChange")

    mode_map = {"1": "zero_shot", "2": "cross_lingual", "3": "instruct", "4": "voice_change"}
    mode = input("\n请输入模式编号: ").strip()
    mode = mode_map.get(mode)

    if mode:
        send_request(mode)
    else:
        print("❌ 无效选择，程序退出！")