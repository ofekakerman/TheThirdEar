import requests


class EmotionAnalysis:
    def __init__(self, hf_key: str):
        self.hf_key = hf_key
        self.url = "https://api-inference.huggingface.co/models/superb/hubert-large-superb-er"

    def get_emotion(self, audio_file_path: str) -> str:
        raw_emotion = self.get_raw_emotion(audio_file_path)
        emotion = self.parse_emotion(raw_emotion)
        return emotion

    def get_raw_emotion(self, audio_file_path: str) -> str:
        headers = {"Authorization": f"Bearer {self.hf_key}"}
        with open(audio_file_path, "rb") as f:
            data = f.read()
        response = requests.post(self.url, headers=headers, data=data)
        emotion = response.json()[0]['label']
        return emotion

    def parse_emotions(self, raw_emotion: str) -> str:
        emotions_dict = {'neu': 'נטרלי',
                         'hap': 'שמח',
                         'sad': 'עצוב',
                         'ang': 'כועס'}
        return emotions_dict[raw_emotion]
