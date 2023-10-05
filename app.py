import gradio as gr
import openai
import tiktoken
import os
from io import BytesIO
import tempfile
import pydub
import re

#トークン上限の設定
max_tokens=4000

def create_meeting_summary(openai_key, uploaded_audio):
    openai.api_key = openai_key
    transcript_text = ""
    audio_size = os.path.getsize(uploaded_audio)
    print(audio_size)
    if audio_size < 26214400:
        print("音声ファイルのサイズが小さいため、そのまま文字起こしを行います。")
        transcript = openai.Audio.transcribe("whisper-1", open(uploaded_audio, "rb"), response_format="verbose_json")
        for segment in transcript.segments:
            transcript_text += f"{segment['text']}\n"
    else:
        print("音声ファイルのサイズが大きいため、分割して文字起こしを行います。")
        audio = pydub.AudioSegment.from_file(uploaded_audio, format="wav")
        audio_duration = audio.duration_seconds
        split_size = 25165824
        split_duration = split_size / (audio_size / audio_duration)
        start = 0
        end = split_duration
        while start < audio_duration:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                audio_segment = audio[start * 1000 : end * 1000] 
                audio_segment.export(temp_file, format="wav") 
                transcript = openai.Audio.transcribe("whisper-1", open(temp_path, "rb"), response_format="verbose_json")
                for segment in transcript.segments:
                    transcript_text += f"{segment['text']}\n"
                os.remove(temp_path) 
            start += split_duration 
            end += split_duration 

    system_template = """大学講義の文字起こしが渡されます。
一度落ち着いてから、よく考えて回答してください。
この講義のサマリーを作成してください。すでに要約ができている場合は、そのまま回答してください。また、リアクションペーパーも作成してください。以下のような形式で書いてください。

- 講義の要約
- リアクションペーパー"""
    #tokenのカウントを行う関数
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    # 文字起こしのテキストを約1900文字ごとに分ける関数を定義する
    def split_text(text, max_tokens):
        print("テキストの分割中")
        sentences = re.split(r"([。？！])", text) # 句読点で分割する
        sentences = ["".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)] # 分割した要素を結合する
        chunks = [] # 分割したテキストを格納するリスト
        chunk = "" # 現在のチャンク
        tokens = 0 # 現在のトークン数
        for sentence in sentences: # 文章ごとにループする
            sentence_tokens = num_tokens_from_string(sentence, "cl100k_base")# 文章のトークン数を計算する
            if tokens + sentence_tokens > max_tokens: # トークン数が上限を超える場合
                chunks.append(chunk) # 現在のチャンクをリストに追加する
                chunk = sentence # 新しいチャンクに文章を代入する
                tokens = sentence_tokens # 新しいトークン数に更新する
            else: # トークン数が上限以下の場合
                chunk += sentence # 現在のチャンクに文章を追加する
                tokens += sentence_tokens # トークン数に加算する
        if chunk: # 最後のチャンクが空でない場合
            chunks.append(chunk) # リストに追加する
        return chunks # リストを返す

    # 分割したテキストごとに要約を行う関数を定義する
    def summarize_text(text):
        print("分割したテキストを要約中...")
        summary_template = """以下のテキストを要約してください。"""
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": text}
        ]
    )
        summary = completion.choices[0].message.content
        print("書き起こした文章を要約したもの..."+summary)
        return summary

    # 文字起こしのテキストを分割する
    if num_tokens_from_string(transcript_text, "cl100k_base") <= max_tokens:
        print("テキストの分割がないため、そのまま要約を行います。")
        summary_text = summarize_text(transcript_text)
    else:
        print("テキストの分割を行い、要約をします。")
        transcript_chunks = split_text(transcript_text, max_tokens)
        # 分割したテキストごとに要約を行う
        summary_chunks = [summarize_text(chunk) for chunk in transcript_chunks]
        # 要約されたテキストを結合する
        summary_text = "".join(summary_chunks)

    # サマリーとリアクションペーパーを作成する
    print("リアクションペーパーを作成中...")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": summary_text}
        ]
    )
    summary = completion.choices[0].message.content
    return summary, transcript_text

inputs = [
    gr.Textbox(lines=1, label="openai_key", type="password"),
    gr.Audio(type="filepath", label="音声ファイルをアップロード")
]

outputs = [
    gr.Textbox(label="サマリー"),
    gr.Textbox(label="文字起こし"),
]
app = gr.Interface(
    fn=create_meeting_summary,
    inputs=inputs,
    outputs=outputs,
    title="サマリー生成アプリ",
    description="音声ファイルをアップロードして、要約を作成します。",
)

app.launch(server_port=7860, server_name="0.0.0.0", debug=True, share=False)
