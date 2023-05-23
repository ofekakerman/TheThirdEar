from emotion_analysis import EmotionAnalysis

def create_conversation_object(conversation, emotion):
    conversation = conversation + f" הרגש הבולט בשיחה היה {emotion} "
    conversation_object = {'role': 'user', 'content': conversation}

    return conversation_object


def create_context_object(context):
    system_promt = {'role': 'user', 'content': ' '.join(context)}
    return system_promt


def create_few_shot_train_general_object(examples):
    train_list = []
    for example in examples:
        question_content = example[0]
        question_dict = {"role": "user", "content": question_content}
        train_list.append(question_dict)
        answer_dict = {"role": "assistant", "content": example[1]}
        train_list.append(answer_dict)
    return train_list


def create_few_shot_train_violence_object(examples):
    train_list = []
    for example in examples:
        question_content = 'איזה סוג אלימות מתבטא במשפט הבא?:' + example[0]
        question_dict = {"role": "user", "content": question_content}
        train_list.append(question_dict)
        answer_dict = {"role": "assistant", "content": example[1]}
        train_list.append(answer_dict)
    return train_list


def create_few_shot_train_warning_object(examples):
    train_list = []
    for example in examples:
        question_content = 'איזה תמרור אזהרה מופיע במשפט הבא?:' + example[0]
        question_dict = {"role": "user", "content": question_content}
        train_list.append(question_dict)
        answer_dict = {"role": "assistant", "content": example[1]}
        train_list.append(answer_dict)
    return train_list


def get_promt(system_promt, train_objects, conversation):
    promt = []
    promt.append(system_promt)
    for train_object in train_objects:
        promt.extend(train_object)
    promt.append(conversation)
    return promt
