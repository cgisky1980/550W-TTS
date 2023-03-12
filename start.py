import matplotlib.pyplot as plt

import os
import json
import math
import time
import threading

import scipy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


from scipy.io.wavfile import write
#import argparse

import websocket
from queue import Queue
import pyaudio
import wave
import numpy as np
import random 
import glob
import jieba


class WebsocketClient(object):
    def __init__(self, address, message_callback=None):
        super(WebsocketClient, self).__init__()
        self.address = address
        self.message_callback = None

    def on_message(self, ws, message):
        
         
        try:
            messages = json.loads((message.encode('raw_unicode_escape')).decode())
            if messages.get("type") == "tts":
                queue_text.put(message)
            elif messages.get("type") == "ping":
                self.ws.send("{\"type\":\"pong\"}")

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
        except KeyError:
            print("KeyError!")

    def on_error(self, ws, error):
        print("client error:",error)

    def on_close(self, ws):
        print("### client closed ###")
        self.ws.close()
        self.is_running = False

    def on_open(self, ws):#连上ws后发布登录信息
        self.is_running = True
        self.ws.send("{\"type\":\"login\",\"uid\":\"tts\",\"pwd\":\"tts9102093109\"}") 

    def close_connect(self):
        self.ws.close()
   
    def send_message(self, message):
        try:
            self.ws.send(message)
        except BaseException as err:
            pass
       
    def run(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(self.address,
                              on_message = lambda ws,message: self.on_message(ws, message),
                              on_error = lambda ws, error: self.on_error(ws, error),
                              on_close = lambda ws :self.on_close(ws))
        self.ws.on_open = lambda ws: self.on_open(ws)
        self.is_running = False
        while True:
            print(self.is_running)
            if not self.is_running:
                self.ws.run_forever()
            time.sleep(3)

class WSClient(object):
    def __init__(self, address, call_back):
        super(WSClient, self).__init__()
        self.client = WebsocketClient(address, call_back)
        self.client_thread = None

    def run(self):
        self.client_thread = threading.Thread(target=self.run_client)
        self.client_thread.start()

    def run_client(self):
        self.client.run()

    def send_message(self, message):
        self.client.send_message(message)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_wav(input_text="哈哈",input_character=0,file="example.wav"):
    global speak_data,tts_working
    stn_tst = get_text(input_text, hps)
    tts_working = True
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid=torch.LongTensor([input_character])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, sid = sid, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        scipy.io.wavfile.write(file, hps.data.sampling_rate , audio)

    speak_data.put(audio.tobytes())

    tts_working = False
        
def get_spk(input_text="哈哈",input_character=0):
    global speak_data,tts_working
    tts_working = True
    print("-----------------tts_working begin!")
    stn_tst = get_text(input_text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid=torch.LongTensor([input_character])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, sid = sid, noise_scale_w=0.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()
        speak_data.put(audio.tobytes())

    tts_working = False
    print("-----------------tts_working end!")
    


def play_wait():
    global spk_working,tts_working,audio_data,speak_data
    while True:
        if  tts_working and not spk_working and  speak_data.empty():
            index = random.randint(0, len(audio_data) - 1)
            selected_data = audio_data[index]
            speak_data.put(selected_data)
            print("-----------------add play_wait!")
 
        time.sleep(10)
    # close PyAudio
    #p.terminate()

def get_queue():
    global queue_text,jieba_text
    while True:
        while not queue_text.empty():
            messages = json.loads((queue_text.get().encode('raw_unicode_escape')).decode())
            queue_text.task_done()
            texts = messages.get("text")
            character = int(messages.get("character"))
            if 'file' in messages: # 判断'file'是否为字典的key
                file = messages.get("file")
            else:
                file = "example.wav"
            if (file == "example.wav"): #如果是不要保存的则进行文字分段
                words = jieba.cut(texts, cut_all=False) # 使用jieba库对文本进行中文分词
                segment = "" # 创建一个空字符串存放当前段落
                for word in words: # 遍历每个词
                    if  len(segment) > 20 and (word == "。" or word == "." or word == "！"  or word == "!" or word == ";" or word == "；" or word == "，" or word == ","): # 如果当前段落长度超过20并且下一个词是句号
                        print("##-----------###")
                        print(segment)
                        jieba_text.put(segment + word)  # 把当前段落和下一个词"."加入分段结果队列
                        segment = "" # 清空当前段落
                    else: # 否则
                        segment += word # 把词加入当前段落
                print(segment)
                jieba_text.put(segment) 
                while not jieba_text.empty():
                    data = jieba_text.get()
                    jieba_text.task_done()
                    get_spk(data, character)
            else:   #如果是要保存的文件则不分段
                get_wav(texts, character,file)


def play_audio(audio_data, sample_rate):
    global spk_working
    spk_working = True
    print("##----spk--start-----###")
    chunk_size = 1024
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True,frames_per_buffer=1024)
    #stream.write(audio_data.tobytes())
    selected_data = audio_data
    for i in range(0, len(selected_data), chunk_size):
        chunk = selected_data[i:i+chunk_size]
        stream.write(chunk)
        while stream.get_write_available() < 1024:
            time.sleep(0.01)
    
    stream.stop_stream()
    stream.close()
    spk_working = False
    print("##----spk--end-----###")
    #p.terminate()

p = pyaudio.PyAudio()
torch.set_num_threads(4)  #RK3588 用4线程效率最佳
# 创建三个队列
queue_text = Queue()    #接收ws信息队列
jieba_text = Queue()    #长文本分词队列
speak_data = Queue()    #要播放的语音数组队列

spk_working = False
tts_working = False
# 读取三个WAV文件的路径
# 读取wav文件夹下所有.wav文件到一个数组
folder = "wavs" # 可以根据需要修改
wav_files = glob.glob(folder + "/*.wav")

# 创建一个空数组来存储读取到的数据
audio_data = []

for filename in wav_files:
    w = wave.open(filename, "r")
    d = w.readframes(w.getnframes())
    audio_data.append(d)
    w.close()
    # read data

hps = utils.get_hparams_from_file("./ys/ys.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,#
    **hps.model)
_ = net_g.eval()
_ = utils.load_checkpoint("./ys/ys.pth", net_g, None)


ws_client = WSClient("ws://localhost:7272", None)
ws_client.run()
gq = threading.Thread(target=get_queue)
gq.start()

pw = threading.Thread(target=play_wait)
pw.start()

while True:
    while not speak_data.empty():
        speak_d = speak_data.get()
        speak_data.task_done()

        play_audio(speak_d, 22050)
        
        time.sleep(1)
    time.sleep(0.5)
