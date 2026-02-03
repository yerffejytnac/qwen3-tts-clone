"""
Gradio demo app for Jeffrey's voice model on Hugging Face Spaces
Web UI for realtime text-to-speech generation with ZeroGPU support
"""

import os
import pickle
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
import spaces
import torch
from qwen_tts import Qwen3TTSModel


class VoiceModelApp:
    def __init__(self) -> None:
        # ZeroGPU will provide CUDA
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self.voice_model_path: str = "models/jeffrey_voice.pkl"
        self.tts: Optional[Qwen3TTSModel] = None
        self.voice_model: Optional[Any] = None

        self.load_models()

    def load_models(self) -> None:
        print(f"Loading TTS model on {self.device}...")
        self.tts = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        print("TTS model loaded")

        print(f"Loading voice model: {self.voice_model_path}")
        with open(self.voice_model_path, "rb") as f:
            self.voice_model = pickle.load(f)
        print("Voice model loaded\n")

    @spaces.GPU(duration=120)
    def generate_speech(
        self,
        text: str,
        language: str,
        x_vector_only: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        subtalker_temperature: float,
        subtalker_top_p: float,
        subtalker_top_k: int,
    ) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
        if not text.strip():
            return None, "Please enter some text"

        if self.tts is None or self.voice_model is None:
            return None, "Models not loaded"

        try:
            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": True,
                "top_k": int(top_k),
                "top_p": float(top_p),
                "temperature": float(temperature),
                "repetition_penalty": float(repetition_penalty),
                "subtalker_dosample": True,
                "subtalker_top_k": int(subtalker_top_k),
                "subtalker_top_p": float(subtalker_top_p),
                "subtalker_temperature": float(subtalker_temperature),
                "non_streaming_mode": False,
            }

            wavs, sr = self.tts.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=self.voice_model,
                x_vector_only_mode=x_vector_only,
                **gen_kwargs,
            )

            # Convert to format Gradio expects
            audio_data = (sr, wavs[0])
            return audio_data, "✓ Generated successfully"

        except Exception as e:
            return None, f"Error: {str(e)}"


def create_demo() -> gr.Blocks:
    app = VoiceModelApp()

    with gr.Blocks(title="Qwen3 TTS Demo") as demo:
        gr.Markdown("# Qwen3 TTS Demo")
        gr.Markdown(f"Checkpoint: `{app.model_path}`")
        gr.Markdown("Model Type: `base`")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here...",
                    lines=5,
                )

                language = gr.Dropdown(
                    choices=[
                        "English",
                        "Chinese",
                        "Japanese",
                        "Korean",
                        "French",
                        "German",
                        "Spanish",
                    ],
                    value="English",
                    label="Language",
                )

                x_vector_only = gr.Checkbox(
                    label="X-vector only mode (faster, lower quality)",
                    value=False,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=4096,
                        value=2048,
                        step=256,
                        label="Max New Tokens",
                        info="Maximum output length",
                    )

                    gr.Markdown("### Main Sampling")
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.9,
                        step=0.1,
                        label="Temperature",
                        info="Randomness (0.1-1.5, higher = more varied)",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Top P",
                        info="Nucleus sampling threshold",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top K",
                        info="Sampling diversity control",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.05,
                        step=0.05,
                        label="Repetition Penalty",
                        info="Reduces repetition (>1.0)",
                    )

                    gr.Markdown("### Subtalker Sampling")
                    gr.Markdown("*Subtalker-specific parameters for improved quality*")
                    subtalker_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.9,
                        step=0.1,
                        label="Subtalker Temperature",
                        info="Subtalker randomness (0.1-1.5, higher = more varied)",
                    )
                    subtalker_top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Subtalker Top P",
                        info="Subtalker nucleus sampling threshold",
                    )
                    subtalker_top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Subtalker Top K",
                        info="Subtalker sampling diversity control",
                    )

                generate_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="numpy",
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        # Examples
        gr.Examples(
            examples=[
                [
                    "I'm not the pheasant plucker, I'm the pheasant plucker's mate. I'm only plucking pheasants 'cause the pheasant plucker's running late"
                ],
                ["I slit the sheet, the sheet I slit, and on the slitted sheet I sit"],
                ["Six sick hicks nick six slick bricks with picks and sticks"],
                [
                    "Extremely accurate and stunningly beautiful bespoke printer profiles transform creative print making to an extraordinary extent."
                ],
                [
                    "Generating code from AI prompts can lead to verbose code, or duplication of existing code instead of using an abstraction. But there are times when this is perfectly acceptable, such as when building proof of concepts, or when topics like program efficiency are unimportant."
                ],
                [
                    "The popularity of modern weight-loss treatments, combined with their high costs and shortages of supplies, have made unofficial forms of them popular in such underground online drug bazaars."
                ],
            ],
            inputs=text_input,
        )

        generate_btn.click(
            fn=app.generate_speech,
            inputs=[
                text_input,
                language,
                x_vector_only,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                subtalker_temperature,
                subtalker_top_p,
                subtalker_top_k,
            ],
            outputs=[audio_output, status_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
