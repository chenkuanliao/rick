import tempfile
from pathlib import Path

import gradio as gr
import librosa
import soundfile as sf
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS


MODEL = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> ChatterboxTurboTTS:
    global MODEL
    if MODEL is None:
        MODEL = ChatterboxTurboTTS.from_pretrained(device=get_device())
    return MODEL


def synthesize(
    text: str,
    audio_prompt_path: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool,
) -> str:
    model = get_model()
    prompt_path = normalize_prompt_audio(audio_prompt_path) if audio_prompt_path else None
    wav = model.generate(
        text=text,
        audio_prompt_path=prompt_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        norm_loudness=norm_loudness,
    )
    audio = wav.squeeze().detach().cpu().numpy()
    output_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(output_path, audio, model.sr)
    return str(output_path)


def normalize_prompt_audio(audio_prompt_path: str) -> str:
    audio, sample_rate = librosa.load(audio_prompt_path, sr=None, mono=True)
    output_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(output_path, audio.astype("float32"), sample_rate)
    return str(output_path)


with gr.Blocks(title="Chatterbox Turbo TTS") as demo:
    gr.Markdown("# Chatterbox Turbo TTS")
    text = gr.Textbox(
        label="Text",
        lines=5,
        value="Hello. This is Chatterbox Turbo running locally.",
    )
    audio_prompt = gr.Audio(
        label="Voice prompt",
        type="filepath",
        sources=["upload", "microphone"],
    )
    with gr.Row():
        temperature = gr.Slider(0.05, 2.0, value=0.8, step=0.05, label="Temperature")
        top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label="Top P")
        repetition_penalty = gr.Slider(
            1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty"
        )
    with gr.Accordion("Advanced", open=False):
        top_k = gr.Slider(1, 2000, value=1000, step=1, label="Top K")
        norm_loudness = gr.Checkbox(value=True, label="Normalize prompt loudness")
    generate = gr.Button("Generate", variant="primary")
    output = gr.Audio(label="Output", type="filepath")

    generate.click(
        synthesize,
        inputs=[
            text,
            audio_prompt,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
