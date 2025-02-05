import grpc
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import sounddevice as sd
import numpy as np
import sys

# 服务器地址（修改为实际 IP）
SERVER_ADDRESS = "localhost:50000"

# 采样率
SAMPLE_RATE = 16000

def play_audio_stream(response):
    """实时播放服务器返回的音频流"""
    audio_buffer = bytearray()

    for res in response:
        audio_buffer.extend(res.tts_audio)

        # 解码为 int16 PCM
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
        if len(audio_array) > 0:
            sd.play(audio_array, samplerate=SAMPLE_RATE)
            sd.wait()
            audio_buffer.clear()  # 清空 buffer，避免重复播放

def send_request(mode):
    """发送请求到服务器，并播放返回的 TTS 音频"""
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

    while True:
        text = input("\n请输入要转换的文本 (输入 'exit' 退出): ").strip()
        if text.lower() == "exit":
            break

        if mode == "sft":
            spk_id = input("请输入说话人ID: ").strip()
            request = cosyvoice_pb2.Request(sft_request=cosyvoice_pb2.SFTRequest(tts_text=text, spk_id=spk_id))
        elif mode == "zero_shot":
            prompt_text = input("请输入音色参考文本 (server 内部查找对应音频): ").strip()
            request = cosyvoice_pb2.Request(zero_shot_request=cosyvoice_pb2.ZeroShotRequest(tts_text=text, prompt_text=prompt_text))
        elif mode == "cross_lingual":
            prompt_text = input("请输入跨语言参考文本 (server 内部查找对应音频): ").strip()
            request = cosyvoice_pb2.Request(cross_lingual_request=cosyvoice_pb2.CrossLingualRequest(tts_text=text, prompt_text=prompt_text))
        elif mode == "instruct":
            spk_id = input("请输入说话人ID: ").strip()
            instruct_text = input("请输入指令文本 (e.g., '用四川话说这句话'): ").strip()
            request = cosyvoice_pb2.Request(instruct_request=cosyvoice_pb2.InstructRequest(tts_text=text, spk_id=spk_id, instruct_text=instruct_text))
        else:
            print("模式错误！")
            return
        
        print("\n🎙️ 服务器处理中...")
        response = stub.Inference(request)
        play_audio_stream(response)

if __name__ == "__main__":
    print("\n请选择模式: ")
    print("1️⃣  SFT (选择固定音色)")
    print("2️⃣  Zero-shot (用文本匹配音色)")
    print("3️⃣  Cross-lingual (跨语言)")
    print("4️⃣  Instruct (指定风格/口音)")

    mode_map = {"1": "sft", "2": "zero_shot", "3": "cross_lingual", "4": "instruct"}
    mode = input("\n请输入模式编号: ").strip()
    mode = mode_map.get(mode)

    if mode:
        send_request(mode)
    else:
        print("❌ 无效选择，程序退出！")