import os
import time

from pyannote.core import Segment
from pydub import AudioSegment, audio_segment

from emotion_analysis import EmotionAnalysis
from speech_segmentation import SpeechSegmentationManager
from stt import Whisper


class ConversationSTT:
    def __init__(self, openai_key: str, hf_key: str):
        self.whisper = Whisper(openai_key)
        self.speech_segmentation_manager = SpeechSegmentationManager(hf_key)
        self.emotion_analysis = EmotionAnalysis(hf_key)

    def audio_conversation_to_text(self, file_dir: str, file_name: str, file_format: str = 'wav',
                                   start_prompt: str = '', max_tries=10, min_segment=1) -> str:
        print('Converting speech to segments')
        file_path = os.path.join(file_dir, f'{file_name}.{file_format}')
        segments = self.speech_segmentation_manager.segment_audio(file_path)
        conversation = AudioSegment.from_wav(file_path)
        tries = 0
        start_prompt = start_prompt + 'זה טקסט השיחה שנאמר עד כה:'
        text_conversation = ""
        full_conversation = ""
        if not os.path.exists(os.path.join(file_dir, file_name)):
            os.mkdir(os.path.join(file_dir, file_name))
        # TODO: Improving the segmentation using the STT.
        #  When there is a long section with interruptions we can split it using overlapping stt
        for turn, _, speaker in segments.itertracks(yield_label=True):
            if turn.end - turn.start > min_segment:
                segment_path = self.extract_segmant(conversation, file_dir, file_format, file_name, turn)
                prompt = start_prompt + '\n' + text_conversation
                while tries < max_tries:
                    try:
                        transcribe = self.whisper.convert_speech_to_text(segment_path, prompt=prompt)
                        break
                    except Exception as e:
                        print(f'Received {e}')
                        print('failed to excess whisper, sleeping for 15 seconds')
                        time.sleep(15)
                try:
                    emotion = self.emotion_analysis.get_emotion(segment_path)
                except Exception as e:
                    print(e)
                    emotion = 'לא ניתן לקבוע רגש'
                text_conversation += transcribe
                full_conversation += f'{turn} {speaker} \n {transcribe}'
                print(turn, speaker)
                print(transcribe)
                print(emotion)
        return full_conversation

    def extract_segmant(self, conversation: audio_segment, file_dir: str, file_format: str, file_name: str,
                        turn: Segment) -> str:
        segment = conversation[int(turn.start * 1000):int(turn.end * 1000)]
        segment_path = os.path.join(file_dir, file_name, f'{file_name}_{turn.start}_{turn.end}.wav')
        segment.export(segment_path, format=file_format)
        return segment_path
