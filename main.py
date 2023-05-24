import os

from dotenv import load_dotenv

from conversation_stt import ConversationSTT


def main():
    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    hf_key = os.getenv('HF_KEY')
    start_prompt = 'זוהי הקלטה של מישהו לחבר שלו בוואטסאפ.'
    conversation_stt = ConversationSTT(openai_key, hf_key)
    conversation_text = conversation_stt.audio_conversation_to_text('Audio', 'shnitzels_80_shekels',
                                                                    start_prompt=start_prompt)
    print('full conversation text')
    print(conversation_text)


if __name__ == '__main__':
    main()
