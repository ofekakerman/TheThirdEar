import os

from dotenv import load_dotenv

from conversation_stt import ConversationSTT


def main():
    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    hf_key = os.getenv('HF_KEY')
    start_prompt = 'זוהי שיחת עימות בין אב אלים לבתו.'
    conversation_stt = ConversationSTT(openai_key, hf_key)
    conversation_text = conversation_stt.audio_conversation_to_text('Audio', 'keren_dad_conversation',
                                                                    start_prompt=start_prompt)
    print('full conversation text')
    print(conversation_text)


if __name__ == '__main__':
    main()
