from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation


class SpeechSegmentationManager:
    def __init__(self, hf_key: str):
        self.hf_key = hf_key
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                 use_auth_token=self.hf_key)

    def segment_audio(self, audio_file: str, max_speakers: int = 2, save_file=None) -> Annotation:
        segments: Annotation = self.pipeline(audio_file, max_speakers=max_speakers)
        if save_file is not None:
            with open(save_file, "w") as rttm:
                segments.write_rttm(rttm)
        return segments
