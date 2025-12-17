# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import sys
import argparse
import numpy as np
import torch
import torchaudio
import random
import librosa
import base64
import io
import gradio as gr
import logging
from typing import Optional, Tuple, Generator
from scipy.io.wavfile import write
import datetime
import time

# 配置日志级别 - 可选择: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level=logging.INFO,  # 修改这里来设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置第三方库的日志级别
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('gradio').setLevel(logging.WARNING)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/CosyVoice'.format(ROOT_DIR))
sys.path.append('{}/CosyVoice/third_party/Matcha-TTS'.format(ROOT_DIR))
from CosyVoice.cosyvoice.cli.cosyvoice import AutoModel
from CosyVoice.cosyvoice.utils.file_utils import load_wav
from CosyVoice.cosyvoice.utils.common import set_all_random_seed

# 全局变量
cosyvoice = None
max_val = 0.8
prompt_sr = 16000
output_dir = "outputs"  # 默认输出目录

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """后处理生成的音频"""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def numpy_to_mp3_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """
    将numpy音频数组转换为MP3字节流，用于Gradio流式音频
    
    Args:
        audio_array: 音频数据数组
        sample_rate: 采样率
        
    Returns:
        MP3格式的字节流
    """
    # 确保音频数据在正确的范围内
    if audio_array.dtype != np.int16:
        # 将float音频转换为int16
        audio_array = (audio_array * 32767).astype(np.int16)
    
    # 创建内存缓冲区
    buffer = io.BytesIO()
    
    # 写入WAV格式到缓冲区
    write(buffer, sample_rate, audio_array)
    
    # 获取字节数据
    buffer.seek(0)
    audio_bytes = buffer.getvalue()
    
    return audio_bytes

def generate_unique_filename(base_name: str, extension: str = ".wav") -> str:
    """
    生成唯一的文件名，避免重复
    
    Args:
        base_name: 基础文件名
        extension: 文件扩展名
        
    Returns:
        唯一的文件名
    """
    global output_dir
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用时间戳生成唯一文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    microseconds = int(time.time() * 1000000) % 1000000  # 微秒精度
    
    filename = f"{base_name}_{timestamp}_{microseconds:06d}{extension}"
    filepath = os.path.join(output_dir, filename)
    
    # 如果文件仍然存在（极小概率），添加计数器
    counter = 1
    while os.path.exists(filepath):
        filename = f"{base_name}_{timestamp}_{microseconds:06d}_{counter}{extension}"
        filepath = os.path.join(output_dir, filename)
        counter += 1
    
    return filepath

def save_audio_to_file(audio_data: Tuple[int, np.ndarray], mode: str, tts_text: str) -> str:
    """
    保存音频数据到文件
    
    Args:
        audio_data: (sample_rate, audio_array) 音频数据
        mode: 推理模式
        tts_text: 合成的文本（不再用于文件名）
        
    Returns:
        保存的文件路径
    """
    sample_rate, audio_array = audio_data
    
    # 根据模式创建基础文件名（只包含模式，不包含文本）
    mode_map = {
        "3s极速复刻": "zero_shot",
        "跨语种复刻": "cross_lingual", 
        "自然语言控制": "instruct"
    }
    mode_short = mode_map.get(mode, "unknown")
    
    base_name = f"cosyvoice_{mode_short}"
    
    # 生成唯一文件路径
    filepath = generate_unique_filename(base_name)
    
    # 确保音频数据格式正确
    if audio_array.dtype != np.int16:
        # 将float音频转换为int16
        audio_array = (audio_array * 32767).astype(np.int16)
    
    # 保存音频文件
    write(filepath, sample_rate, audio_array)
    
    logging.info(f"音频已保存到: {filepath}")
    return filepath

def generate_audio(
    tts_text: str,
    prompt_audio: Optional[str],
    prompt_text: str,
    mode: str,
    instruct_text: str = "",
    seed: Optional[int] = None,
    speed: float = 1.0,
    streaming: bool = False
) -> Tuple[Optional[Tuple], str]:
    """
    生成音频的主函数
    
    Args:
        tts_text: 要合成的文本
        prompt_audio: 提示音频文件路径
        prompt_text: 提示文本
        mode: 推理模式
        instruct_text: 指令文本（自然语言控制模式使用）
        seed: 随机种子
        speed: 语速控制
        streaming: 是否使用流式生成
        
    Returns:
        Tuple[audio_tuple, message]
        audio_tuple: (sample_rate, audio_array) 或 None
        message: 状态信息字符串
    """
    global cosyvoice
    
    try:
        # 验证输入
        if not tts_text or tts_text.strip() == "":
            return None, "❌ 错误: 请输入要合成的文本"
            
        if not prompt_audio:
            return None, "❌ 错误: 请上传提示音频文件"
            
        # 检查prompt文本
        if prompt_text.strip() == "" and mode in ['3s极速复刻', '自然语言控制', '跨语种复刻']:
            return None, "❌ 错误: 该模式需要提供提示文本"
            
        # 检查指令文本
        if mode == '自然语言控制' and instruct_text.strip() == "":
            return None, "❌ 错误: 自然语言控制模式需要提供指令文本"
        
        # 检查音频采样率
        audio_info = torchaudio.info(prompt_audio)
        if audio_info.sample_rate < prompt_sr:
            return None, f"❌ 错误: 提示音频采样率 {audio_info.sample_rate} 低于要求的 {prompt_sr} Hz"
        
        # 设置随机种子
        if seed is None:
            seed = random.randint(1, 100000000)
        set_all_random_seed(seed)
        
        result_audio = None
        
        # 根据模式进行推理 - 直接传递音频文件路径
        if mode == '3s极速复刻':
            logging.info('执行零样本推理')
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_audio, stream=streaming, speed=speed):
                audio = i['tts_speech'].numpy().flatten()
                if result_audio is None:
                    result_audio = audio
                else:
                    result_audio = np.concatenate([result_audio, audio])
                    
        elif mode == '跨语种复刻':
            logging.info('执行跨语言推理')
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_audio, stream=streaming, speed=speed):
                audio = i['tts_speech'].numpy().flatten()
                if result_audio is None:
                    result_audio = audio
                else:
                    result_audio = np.concatenate([result_audio, audio])
                    
        elif mode == '自然语言控制':
            logging.info('执行指令推理')
            for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_audio, stream=streaming, speed=speed):
                audio = i['tts_speech'].numpy().flatten()
                if result_audio is None:
                    result_audio = audio
                else:
                    result_audio = np.concatenate([result_audio, audio])
        
        if result_audio is not None:
            audio_data = (cosyvoice.sample_rate, result_audio)
            
            # 自动保存音频文件
            try:
                saved_path = save_audio_to_file(audio_data, mode, tts_text)
                return audio_data, f"✅ 音频生成成功！使用种子: {seed}\n💾 已保存到: {saved_path}"
            except Exception as e:
                logging.warning(f"保存音频文件失败: {e}")
                return audio_data, f"✅ 音频生成成功！使用种子: {seed}\n⚠️ 保存失败: {str(e)}"
        else:
            return None, "❌ 错误: 音频生成失败"
            
    except Exception as e:
        error_msg = f"❌ 生成音频时发生错误: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

def generate_audio_streaming_with_complete(
    tts_text: str,
    prompt_audio: Optional[str],
    prompt_text: str,
    mode: str,
    instruct_text: str = "",
    seed: Optional[int] = None,
    speed: float = 1.0
) -> Generator[Tuple[Optional[bytes], Optional[Tuple], str], None, None]:
    """
    改进的流式音频生成函数 - 同时支持流式播放和完整音频
    
    Args:
        tts_text: 要合成的文本
        prompt_audio: 提示音频文件路径
        prompt_text: 提示文本
        mode: 推理模式
        instruct_text: 指令文本（自然语言控制模式使用）
        seed: 随机种子
        speed: 语速控制
        
    Yields:
        Tuple[streaming_bytes, complete_audio, message]: 
        - streaming_bytes: 流式播放的音频片段字节流
        - complete_audio: 完整音频(sample_rate, audio_array)或None
        - message: 状态信息
    """
    global cosyvoice
    
    try:
        # 验证输入
        if not tts_text or tts_text.strip() == "":
            yield None, None, "❌ 错误: 请输入要合成的文本"
            return
            
        if not prompt_audio:
            yield None, None, "❌ 错误: 请上传提示音频文件"
            return
            
        # 检查prompt文本
        if prompt_text.strip() == "" and mode in ['3s极速复刻', '自然语言控制', '跨语种复刻']:
            yield None, None, "❌ 错误: 该模式需要提供提示文本"
            return
            
        # 检查指令文本
        if mode == '自然语言控制' and instruct_text.strip() == "":
            yield None, None, "❌ 错误: 自然语言控制模式需要提供指令文本"
            return
        
        # 检查音频采样率
        audio_info = torchaudio.info(prompt_audio)
        if audio_info.sample_rate < prompt_sr:
            yield None, None, f"❌ 错误: 提示音频采样率 {audio_info.sample_rate} 低于要求的 {prompt_sr} Hz"
            return
        
        # 设置随机种子
        if seed is None:
            seed = random.randint(1, 100000000)
        set_all_random_seed(seed)
        
        chunk_count = 0
        accumulated_audio = None
        
        # 根据模式进行流式推理 - 直接传递音频文件路径
        inference_generator = None
        if mode == '3s极速复刻':
            logging.info('执行零样本流式推理')
            inference_generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_audio, stream=True, speed=speed)
                    
        elif mode == '跨语种复刻':
            logging.info('执行跨语言流式推理')
            inference_generator = cosyvoice.inference_cross_lingual(tts_text, prompt_audio, stream=True, speed=speed)
                    
        elif mode == '自然语言控制':
            logging.info('执行指令流式推理')
            inference_generator = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_audio, stream=True, speed=speed)
        
        if inference_generator:
            for i in inference_generator:
                audio_chunk = i['tts_speech'].numpy().flatten()
                chunk_count += 1
                
                # 累积音频片段用于完整音频
                if accumulated_audio is None:
                    accumulated_audio = audio_chunk
                else:
                    accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])
                
                # 转换音频片段为字节流（用于流式播放）
                audio_bytes = numpy_to_mp3_bytes(audio_chunk, cosyvoice.sample_rate)
                
                # yield流式音频片段，完整音频还没准备好
                yield audio_bytes, None, f"🔊 正在生成... 片段 {chunk_count} (种子: {seed})"
        
        # 最终完成 - 提供完整的累积音频并自动保存
        if accumulated_audio is not None:
            complete_audio = (cosyvoice.sample_rate, accumulated_audio)
            
            # 自动保存完整音频文件
            try:
                saved_path = save_audio_to_file(complete_audio, mode, tts_text)
                yield None, complete_audio, f"✅ 流式音频生成完成！共 {chunk_count} 个片段 (种子: {seed})\n💾 已保存到: {saved_path}"
            except Exception as e:
                logging.warning(f"保存音频文件失败: {e}")
                yield None, complete_audio, f"✅ 流式音频生成完成！共 {chunk_count} 个片段 (种子: {seed})\n⚠️ 保存失败: {str(e)}"
        else:
            yield None, None, "❌ 错误: 流式音频生成失败"
            
    except Exception as e:
        error_msg = f"❌ 流式生成音频时发生错误: {str(e)}"
        logging.error(error_msg)
        yield None, None, error_msg

def generate_audio_streaming(
    tts_text: str,
    prompt_audio: Optional[str],
    prompt_text: str,
    mode: str,
    instruct_text: str = "",
    seed: Optional[int] = None,
    speed: float = 1.0
) -> Generator[Tuple[bytes, str], None, None]:
    """保留原有的纯流式函数以便兼容"""
    for streaming_bytes, complete_audio, message in generate_audio_streaming_with_complete(
        tts_text, prompt_audio, prompt_text, mode, instruct_text, seed, speed
    ):
        if streaming_bytes is not None:
            yield streaming_bytes, message

def create_gradio_interface():
    """创建 Gradio 界面"""
    
    # 自定义 CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2563eb;
        font-weight: bold;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="CosyVoice 语音合成", theme=gr.themes.Soft()) as interface:
        
        # 标题和说明
        gr.HTML("""
        <div class="main-header">
            <h1>🎵 CosyVoice 语音合成系统</h1>
            <p>基于深度学习的多模态语音合成，支持零样本、跨语言和指令控制</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.HTML('<h3 class="section-header">📝 输入设置</h3>')
                
                tts_text = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文本内容...",
                    lines=3,
                    max_lines=5
                )
                
                prompt_audio = gr.Audio(
                    label="提示音频 (上传参考音频，用于声音克隆)",
                    type="filepath"
                )
                
                prompt_text = gr.Textbox(
                    label="提示文本 (提示音频对应的文字内容，零样本和跨语言模式必填)",
                    placeholder="输入提示音频对应的文本内容...",
                    lines=2
                )
                
                mode = gr.Dropdown(
                    choices=["3s极速复刻", "跨语种复刻", "自然语言控制"],
                    value="3s极速复刻",
                    label="推理模式 (选择语音合成模式)"
                )
                
                instruct_text = gr.Textbox(
                    label="指令文本 (自然语言控制模式的指令，仅在该模式下显示)",
                    placeholder="例如：请用温柔的语调朗读...",
                    lines=2,
                    visible=False
                )
                
                streaming_mode = gr.Checkbox(
                    label="🔊 流式播放模式 (边生成边播放，实时预览)",
                    value=False,
                    interactive=True
                )
                
                # 高级设置
                with gr.Accordion("🔧 高级设置", open=False):
                    with gr.Row():
                        seed = gr.Number(
                            label="随机种子 (控制生成的随机性)",
                            value=None,
                            precision=0
                        )
                        speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="语速 (调节语音播放速度)"
                        )
                
                # 生成按钮
                generate_btn = gr.Button(
                    "🚀 开始生成",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                # 输出区域
                gr.HTML('<h3 class="section-header">🔊 生成结果</h3>')
                
                # 流式音频组件（实时播放片段）
                streaming_audio = gr.Audio(
                    label="🎵 流式播放（实时）",
                    interactive=False,
                    streaming=True,  # 启用流式音频支持
                    autoplay=True,   # 自动播放新的音频片段
                    visible=False    # 默认不显示，根据模式切换
                )
                
                # 普通音频组件（完整音频）
                output_audio = gr.Audio(
                    label="📄 完整音频",
                    interactive=False,
                    streaming=False  # 普通音频组件
                )
                
                output_message = gr.Textbox(
                    label="状态信息",
                    lines=3,
                    interactive=False
                )
                
                # 预置示例
                with gr.Accordion("📋 使用示例", open=False):
                    gr.HTML("""
                    <div style="padding: 1rem; background-color: #f8fafc; border-radius: 0.5rem;">
                        <h4>使用说明：</h4>
                        <ol>
                            <li><strong>3s极速复刻：</strong> 基于3秒参考音频进行声音克隆</li>
                            <li><strong>跨语种复刻：</strong> 保持音色进行跨语言合成</li>
                            <li><strong>自然语言控制：</strong> 通过自然语言指令控制合成效果</li>
                        </ol>
                        <h4>播放模式：</h4>
                        <ul>
                            <li><strong>🔊 流式播放模式：</strong> 
                                <br>• <strong>🎵 流式播放</strong>：实时播放音频片段，立即听到生成效果
                                <br>• <strong>📄 完整音频</strong>：生成完成后显示完整的可下载音频文件
                            </li>
                            <li><strong>📄 正常模式：</strong> 只显示完整音频，生成完成后一次性播放</li>
                        </ul>
                        <p><strong>💡 双音频优势：</strong> 流式模式下既能实时预览，又能获得完整的音频文件！</p>
                        <p><strong>💾 自动保存：</strong> 所有生成的完整音频都会自动保存到outputs目录，文件名包含模式、文本和时间戳</p>
                        <p><strong>提示：</strong> 请确保上传的音频质量清晰，采样率不低于16kHz</p>
                    </div>
                    """)
        
        # 动态显示指令文本框
        def update_instruct_visibility(mode_value):
            return gr.update(visible=(mode_value == "自然语言控制"))
        
        # 动态显示流式音频组件
        def update_streaming_audio_visibility(streaming_enabled):
            return gr.update(visible=streaming_enabled)
        
        # 统一处理音频生成的函数
        def handle_audio_generation(tts_text, prompt_audio, prompt_text, mode, instruct_text, seed, speed, streaming):
            if streaming:
                # 流式生成模式 - 同时更新流式和普通音频组件
                for streaming_bytes, complete_audio, message in generate_audio_streaming_with_complete(
                    tts_text, prompt_audio, prompt_text, mode, instruct_text, seed, speed
                ):
                    yield streaming_bytes, complete_audio, message  # streaming_audio, output_audio, message
            else:
                # 正常生成模式 - 只更新普通音频组件
                audio_result, message = generate_audio(
                    tts_text, prompt_audio, prompt_text, mode, instruct_text, seed, speed, streaming=False
                )
                yield None, audio_result, message  # streaming_audio, output_audio, message
        
        mode.change(
            fn=update_instruct_visibility,
            inputs=[mode],
            outputs=[instruct_text]
        )
        
        # 流式模式切换事件
        streaming_mode.change(
            fn=update_streaming_audio_visibility,
            inputs=[streaming_mode],
            outputs=[streaming_audio]
        )
        
        # 绑定生成按钮 - 现在输出到三个组件
        generate_btn.click(
            fn=handle_audio_generation,
            inputs=[tts_text, prompt_audio, prompt_text, mode, instruct_text, seed, speed, streaming_mode],
            outputs=[streaming_audio, output_audio, output_message],
            show_progress=True
        )
        
        # 示例输入
        gr.Examples(
            examples=[
                [
                    "今天天气真不错，我们一起去公园走走吧！",
                    None,  # 这里需要用户自己上传音频
                    "你好，欢迎使用语音合成系统。",
                    "3s极速复刻",
                    "",
                    12345,
                    1.0,
                    False  # 默认不使用流式模式
                ],
                [
                    "Hello, welcome to the voice synthesis system.",
                    None,
                    "你好，欢迎使用语音合成系统。",
                    "跨语种复刻", 
                    "",
                    54321,
                    1.0,
                    False
                ],
                [
                    "这是一个测试文本。",
                    None,
                    "参考音频文本",
                    "自然语言控制",
                    "请用温柔甜美的声音朗读",
                    98765,
                    1.0,
                    True  # 展示流式模式
                ]
            ],
            inputs=[tts_text, prompt_audio, prompt_text, mode, instruct_text, seed, speed, streaming_mode],
        )
    
    return interface

def main():
    parser = argparse.ArgumentParser(description='CosyVoice 语音合成 Web 界面')
    parser.add_argument('--port',
                        type=int,
                        default=7860,
                        help='服务端口号')
    parser.add_argument('--host',
                        type=str,
                        default='127.0.0.1',
                        help='服务主机地址')
    parser.add_argument('--model-dir',
                        type=str,
                        default='CosyVoice/pretrained_models/CosyVoice3-0.5B',
                        help='模型路径或 modelscope repo id')
    parser.add_argument('--output-dir',
                        type=str,
                        default='outputs',
                        help='音频文件自动保存目录')
    parser.add_argument('--share',
                        action='store_true',
                        help='创建公共分享链接')
    parser.add_argument('--log-level',
                        type=str,
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志输出级别')
    args = parser.parse_args()
    
    # 根据命令行参数设置日志级别
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    logging.info(f"日志级别设置为: {args.log_level}")
    
    global cosyvoice, output_dir
    
    # 设置输出目录
    output_dir = args.output_dir
    
    # 初始化模型
    print("🔄 正在加载模型...")
    try:
        cosyvoice = AutoModel(model_dir=args.model_dir)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 创建并启动界面
    interface = create_gradio_interface()
    
    print(f"🚀 启动 Web 服务...")
    print(f"📍 访问地址: http://{args.host}:{args.port}")
    print(f"💾 音频保存目录: {os.path.abspath(output_dir)}")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )

if __name__ == '__main__':
    main()
