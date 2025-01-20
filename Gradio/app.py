import base64
import io
import tempfile
from dataclasses import dataclass, field

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from llm import speech_to_text, text_generation, text_to_speech
from openai import OpenAI
from PIL import Image, ImageOps
from pydub import AudioSegment
from silero_vad import load_silero_vad
from vad import determine_pause

load_dotenv()


STREAM_EVERY = 1.0
HOW_TO_USE = (
    "1. 「Input Audio」コンポーネントのRecordボタンをクリックしてください。\n"
    "2. 「Image」コンポーネントをクリックしてWebカメラにアクセスし、Recordボタンをクリックしてください。\n\n"
    "   また、以下のチェックボックスで設定を変更できます。\n\n"
    "   - **Use Image** : LLMの入力にカメラ画像を使用します\n"
    "   - **Flip Image** : カメラ画像を左右反転します\n\n"
    "3. 音声を入力することで、LLMと会話が可能です。発話の開始と終了を自動的に検知して応答します。"
)

vad = load_silero_vad(onnx=False)
client = OpenAI()


@dataclass
class AppState:
    """
    AppState class to manage the state of the application.

    Attributes:
        stream (np.ndarray): An array to store audio stream data.
        sampling_rate (int): The sampling rate of the audio stream.
        pause_detected (bool): A flag to indicate if a pause is detected in the audio stream.
        started_talking (bool): A flag to indicate if talking has started.

        use_image (bool): A flag to indicate if an image is being used.
        flip_image (bool): A flag to indicate if the image should be flipped.
        current_image_path (str): The path to the current stream image.
        image_path (str): The path to the image for LLM input.

        conversation (list): A list to store the conversation history for the Chatbot component input.
        chat_history (list): A list to store the chat history for LLM input.
    """

    # audio
    stream: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool = False

    # image
    use_image: bool = False
    flip_image: bool = False
    current_image_path: str = ""
    image_path: str = ""

    # history
    conversation: list = field(default_factory=list)
    chat_history: list = field(default_factory=list)


def save_image(image_path, state: AppState) -> AppState:
    """
    Saves an image to the specified path, optionally flipping it if the state requires.

    Args:
        image_path (str): The path where the image will be saved.
        state (AppState): The current application state containing image processing options.

    Returns:
        AppState: The updated application state with the current image path set.
    """
    print("save_image", image_path)
    with Image.open(image_path) as im:
        if state.flip_image:
            im = ImageOps.mirror(im)
        im.save(image_path, "PNG")
    print(image_path)
    state.current_image_path = image_path
    return state


def process_audio(
    audio: tuple[int, np.ndarray] | None, state: AppState
) -> tuple[gr.Audio | None, AppState]:
    """
    Processes a single-channel audio input and updates the application state based on detected speech activity.

    Parameters:
        audio (tuple[int, np.ndarray] | None): A tuple containing the sampling rate and the one-dimensional
            audio data array, or None if no audio is provided.
        state (AppState): The current application state holding audio stream information, image paths,
            and flags indicating speech activity.

    Returns:
        tuple[gr.Audio | None, AppState]: A tuple containing:
            - An optional gr.Audio object set to stop recording when a pause is detected.
            - The updated application state with any changes in audio stream, flags, and image path.
    """
    if (state.use_image is True) and (state.current_image_path == ""):
        print("stanby...")
        return gr.skip()

    sampling_rate = audio[0]
    stream = audio[1]
    if stream.ndim == 2:
        stream = stream.mean(axis=1, dtype=np.int16)

    state.sampling_rate = sampling_rate
    state.stream = np.concatenate([state.stream, stream])
    state.image_path = state.current_image_path
    state.pause_detected = determine_pause(vad, stream, state)

    if (not state.pause_detected) and (not state.started_talking):
        print("started talking")
        state.stream = state.stream[-int(sampling_rate * STREAM_EVERY * 2) :]
        state.started_talking = True

    if (state.pause_detected) and (state.started_talking):
        return gr.Audio(recording=False), state

    return gr.skip(), state


def response(state: AppState):
    """
    Handles audio-based interaction by recording, transcribing, and generating responses.

    Args:
        state (AppState): Contains conversation state and audio data.

    Returns:
        tuple:
            - A Gradio audio component indicating ongoing recording.
            - The raw PCM data of synthesized speech content.
            - An updated conversation history containing the user and assistant messages.
            - A new AppState object with updated conversation and chat history.

    Process:
        1. Checks if audio input should be processed based on user's pause and talking state.
        2. Converts recorded audio to WAV format and transcribes it.
        3. Optionally includes an image in the conversation if specified.
        4. Invokes text generation to formulate a response.
        5. Converts the generated text into speech and stores it temporarily.
        6. Appends both text and audio responses to the conversation and chat history.
    """
    if (not state.pause_detected) and (not state.started_talking):
        return gr.skip()

    audio_buffer = io.BytesIO()
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=1,
    )
    segment.export(audio_buffer, format="wav")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
        input_file.write(audio_buffer.getvalue())

    text = speech_to_text(client, input_file.name)
    if state.use_image:
        with open(state.image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        state.conversation.extend(
            [
                {
                    "role": "user",
                    "content": {"path": input_file.name, "mime_type": "audio/wav"},
                },
                {"role": "user", "content": text},
                {
                    "role": "user",
                    "content": {"path": state.image_path, "mime_type": "image/png"},
                },
            ]
        )
        state.chat_history.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        )
    else:
        state.conversation.extend(
            [
                {
                    "role": "user",
                    "content": {"path": input_file.name, "mime_type": "audio/wav"},
                },
                {"role": "user", "content": text},
            ]
        )
        state.chat_history.append(
            {
                "role": "user",
                "content": text,
            }
        )

    assistant_response = text_generation(client, state.chat_history)
    assert len(assistant_response) < 4096, "Response too long"
    speech = text_to_speech(client, assistant_response)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
        speech.write_to_file(output_file.name)

    state.conversation.extend(
        [
            {
                "role": "assistant",
                "content": {"path": output_file.name, "mime_type": "audio/wav"},
            },
            {"role": "assistant", "content": assistant_response},
        ]
    )
    state.chat_history.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )

    ret = (
        gr.Audio(recording=True),
        speech.content,
        state.conversation,
        AppState(
            use_image=state.use_image,
            flip_image=state.flip_image,
            conversation=state.conversation,
            chat_history=state.chat_history,
        ),
    )

    return ret


def toggle_image_button(flag: bool, state: AppState):
    """
    Updates the application state to indicate if an image should be used.

    Parameters:
        flag (bool): Flag that indicates if an image should be used.
        state (AppState): The current application state.

    Returns:
        AppState: The updated application state with the new use_image value set.
    """
    print(f"use_image: {flag}")
    state.use_image = flag
    return state


def toggle_image_flip(flag: bool, state: AppState):
    """
    Toggle the flip_image property in the application state.

    Parameters:
        flag (bool): Indicates whether the image should be flipped.
        state (AppState): The current application state.

    Returns:
        AppState: The updated application state with the flip_image attribute set.
    """
    print(f"flip_image: {flag}")
    state.flip_image = flag
    return state


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("How to use", open=False):
                gr.Markdown(HOW_TO_USE)
            input_audio = gr.Audio(
                label="Input Audio",
                sources=["microphone"],
                type="numpy",
            )
            with gr.Group():
                input_image = gr.Image(
                    label="Image", sources=["webcam"], type="filepath", format="png"
                )
                with gr.Row():
                    image_button = gr.Checkbox(
                        label="Use Image", value=False, container=False
                    )
                    image_flip_button = gr.Checkbox(
                        label="Flip Image", value=False, container=False
                    )
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", type="messages", min_height=800)
            output_audio = gr.Audio(
                label="Output Audio", streaming=False, autoplay=True
            )

    state = gr.State(value=AppState())
    image_button.select(
        toggle_image_button, inputs=[image_button, state], outputs=[state]
    )
    image_flip_button.select(
        toggle_image_flip,
        inputs=[image_flip_button, state],
        outputs=[state],
    )
    input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=STREAM_EVERY,
    )
    input_image.stream(
        save_image,
        [input_image, state],
        [state],
        stream_every=STREAM_EVERY,
    )
    input_audio.stop_recording(
        response,
        [state],
        [input_audio, output_audio, chatbot, state],
        concurrency_limit=1,
    )

demo.launch(share=True)
