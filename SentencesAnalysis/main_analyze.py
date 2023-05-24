import os

import openai
from dotenv import load_dotenv
from SentencesAnalysis.analyze import analyse
from SentencesAnalysis.constants import DEFAULT_CONTEXT, DEFAULT_GENERAL_EXAMPLES, DEFAULT_WARNING_SIGN_EXAMPLES, \
    DEFAULT_VIOLENCE_EXAMPLES, DEFAULT_EMOTION
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
    conversation_object = create_conversation_object(conversation, DEFAULT_EMOTION)
    train_objects = [violence_train_object, warning_train_object, general_train_object]

    prompt = get_promt(system_promt_object, train_objects, conversation_object)
    print ('analyse..')
    ans = analyse(prompt)
    print(ans)
    return ans


if __name__ == "__main__":
    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    openai.api_key=openai_key

    conversation = """
    היי, מה נשמע?
    תקשיבי זה חייב להיפסק
    את לא יכולה להמשיך לבזבז כל כך הרבה כסף
    אם זה ימשיך ככה אני אאלץ להתנקם בך על זה וזה יהיה לא נעים בכלל
    אני לא עומד יותר בעול הכלכלי שאתה גוזרת עלינו
    איך את יכולה להזמין שניצל ב80 שקל שאת יודעת את המצב הכלכלי שלנו
    בושה!
    איך את מסתכלת לעצמך בעיניים?
    ואו ואוו אני רותח מכעס
                        """

    run_analyse(conversation)