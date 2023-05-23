import openai


class Whisper:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        openai.api_key = self.openai_key

    def convert_speech_to_text(self, file_name: str, prompt: str = '', language: str = 'he') -> str:
        audio_file = open(file_name, 'rb')
        transcribe = openai.Audio.transcribe(file=audio_file, model='whisper-1', language=language,
                                             response_format='text',
                                             prompt=prompt)
        return transcribe
