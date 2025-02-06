import grpc
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import sounddevice as sd
import numpy as np
import wave
import sys

# 服务器地址（修改为实际 IP）
SERVER_ADDRESS = "20.0.137.8:50000"

# 采样率
SAMPLE_RATE = 16000

def play_audio(audio_buffer):
    """实时播放音频"""
    audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
    if len(audio_array) > 0:
        sd.play(audio_array, samplerate=SAMPLE_RATE)
        sd.wait()

def save_audio(audio_buffer, output_filename="output_audio.wav"):
    """保存音频为 WAV 文件"""
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_buffer)
    print(f"💾 音频已保存至 {output_filename}")

def play_and_save_audio_stream(response):
    """接收服务器返回的音频流，并分别进行播放和保存"""
    audio_buffer = bytearray()
    
    for res in response:
        audio_buffer.extend(res.tts_audio)
    
    play_audio(audio_buffer)
    save_audio(audio_buffer)

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
            prompt_text = input("请输入跨语言参考文本 (server 内部查找对应音频): ").strip()
            request = cosyvoice_pb2.Request(
                cross_lingual_request=cosyvoice_pb2.CrossLingualRequest(tts_text=text, prompt_text=prompt_text)
            )
        elif mode == "instruct":
            spk_id = input("请输入说话人ID: ").strip()
            instruct_text = input("请输入指令文本 (e.g., '用四川话说这句话'): ").strip()
            request = cosyvoice_pb2.Request(
                instruct_request=cosyvoice_pb2.InstructRequest(tts_text=text, spk_id=spk_id, instruct_text=instruct_text)
            )
        else:
            print("模式错误！")
            return
        
        print("\n🎙️ 服务器处理中...")
        response = stub.Inference(request)
        
        # 播放并保存音频
        play_and_save_audio_stream(response)

if __name__ == "__main__":
    print("\n请选择模式: ")
    print("1 Zero-shot (用文本匹配音色)")
    print("2 Cross-lingual (跨语言)")
    print("3 Instruct (指定风格/口音)")

    mode_map = {"1": "zero_shot", "2": "cross_lingual", "3": "instruct"}
    mode = input("\n请输入模式编号: ").strip()
    mode = mode_map.get(mode)

    if mode:
        send_request(mode)
    else:
        print("❌ 无效选择，程序退出！")