import os

import openai
from dotenv import load_dotenv
from SentencesAnalysis.analyze import analyse
from SentencesAnalysis.constants import DEFAULT_CONTEXT, DEFAULT_GENERAL_EXAMPLES, DEFAULT_WARNING_SIGN_EXAMPLES, \
    DEFAULT_VIOLENCE_EXAMPLES
from SentencesAnalysis.promt_manager import create_few_shot_train_violence_object, create_few_shot_train_warning_object, \
    create_few_shot_train_general_object, create_context_object, create_conversation_object, get_promt

def run_analyse(conversation,
                violence_examples = DEFAULT_VIOLENCE_EXAMPLES,
                warning_signs_examples = DEFAULT_WARNING_SIGN_EXAMPLES,
                general_examples = DEFAULT_GENERAL_EXAMPLES,
                context = DEFAULT_CONTEXT):
    violence_train_object = create_few_shot_train_violence_object(violence_examples)
    warning_train_object = create_few_shot_train_warning_object(warning_signs_examples)
    general_train_object = create_few_shot_train_general_object(general_examples)
    system_promt_object = create_context_object(context)
    conversation_object = create_conversation_object(conversation, 'כעס')
    train_objects = [violence_train_object, warning_train_object, general_train_object]

    promt = get_promt(system_promt_object, train_objects, conversation_object)
    ans = analyse(promt)
    print(ans)
    return ans


if __name__ == "__main__":
    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    openai.api_key=openai_key

    conversation = """.אופק, מה אתה חושב שאתה עושה? תראה לי את הטלפון שלך מיד אני רוצה לראות עם מי אתה מדבר,
      אתה יודע שבן אדם אחר היה רוצח אותך, עוד פעם אחת שזה קורה ואתה לא תראה ממני שקל אתה תעבור לגור ברחוב
    """

    run_analyse(conversation)