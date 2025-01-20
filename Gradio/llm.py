import base64
import functools

import gradio as gr
from openai import OpenAI


def retry_on_error(retries=3):
    """
    Retries the decorated function up to the specified number of times when an exception occurs.

    Parameters:
        retries (int): The maximum number of retries.

    Returns:
        The return value of the decorated function if successful.

    Raises:
        gr.Error: If the final retry fails, the exception is raised with a Gradio error message.

    Example:
        @retry_on_error(retries=3)
        def unstable_operation(...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise gr.Error(f"Failed in {func.__name__}: {e}")
                    else:
                        gr.Warning(f"Attempt {i + 1} failed in {func.__name__}: {e}")

        return wrapper

    return decorator


@retry_on_error()
def text_generation(client: OpenAI, chat_history: list) -> str:
    """
    Generates a text response using the OpenAI API based on the provided chat history.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        chat_history (list): A list of dictionaries representing the chat history.

    Returns:
        str: The generated text response from the model.
    """
    system_prompt = (
        "あなたは親切なアシスタントです．"
        "ユーザからの問い合わせに簡潔に回答してください．"
    )
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    print("tg")
    completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        max_completion_tokens=256,
        temperature=1.0,
    )
    return completion.choices[0].message.content


@retry_on_error()
def speech_to_text(client: OpenAI, audio_file: str, model="gpt") -> str:
    """
    Transcribes speech from an audio file to text using the OpenAI API.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        audio_file (str): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text from the audio file.
    """
    print("stt")
    f = open(audio_file, "rb")
    if model == "gpt":
        wav_data = f.read()
        encoded_string = base64.b64encode(wav_data).decode("utf-8")
        transcript = (
            client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Accurately transcribe the following Japanese audio recording into clear, readable text. If nothing can be understood, return a single space.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
            )
            .choices[0]
            .message.audio.transcript
        )
    elif model == "whisper":
        transcript = client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            language="ja",
            response_format="text",
            temperature=0.0,
        ).strip()
    else:
        raise Exception()
    return transcript


@retry_on_error()
def text_to_speech(client: OpenAI, text: str):
    """
    Converts text to speech using the OpenAI API.

    Args:
        client (OpenAI): An instance of the OpenAI client.
        text (str): The text to be converted to speech.

    Returns:
        openai._legacy_response.HttpxBinaryResponseContent: The generated speech in WAV format.
    """
    print("tts")
    speech = client.audio.speech.create(
        model="tts-1",  # tts-1 or tts-1-hd
        input=text,
        voice="alloy",
        response_format="wav",
        speed=1.0,
    )
    return speech
