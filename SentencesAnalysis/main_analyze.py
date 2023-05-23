import os
from dotenv import load_dotenv
from SentencesAnalysis.analyze import analyse
from SentencesAnalysis.promt_manager import create_few_shot_train_violence_object, create_few_shot_train_warning_object, \
    create_few_shot_train_general_object, create_context_object, create_conversation_object, get_promt

if __name__ == "__main__":
    load_dotenv()
    openai_key = os.getenv('OPENAI_KEY')
    violence_examples = [
        ['"עוד מילה אחת שיוצאת לך מהפה אני לא קונה לך יותר אוכל"', 'אלימות כלכלית'],
        ['"את לא מתביישת לבקש ממני לדבר אלייך בכבוד אחרי שקניתי לך את הטבעת?"', 'אלימות כלכלית']
    ]
    violence_train_object = create_few_shot_train_violence_object(violence_examples)

    warning_signs_examples = [
        ['"מה קרה שהגעת רק עכשיו הביתה? תגידי לי מיד איפה היית!"', 'אובססיביות'],
        ['"את לא יודעת לעשות שום דבר טוב, לא לבשל, לא כלום"', 'הקטנה']
    ]
    warning_train_object = create_few_shot_train_warning_object(warning_signs_examples)

    general_examples = [
        ['"מה קרה? איפה היית אתמול? אני קצת מתבאס שלא ראיתי אותך כל היום"',
         'השיחה תקינה, לא עלו תמרורי אזהרה או סוגי אלימות מובהקים'],
        ['"תשמעי לי טוב, את פשוט אישה רעה את לא טובה בשום דבר אם את ממשיכה להתנהג ככה אני ארצח אותך "',
         'במשפט "את פשוט אישה רעה, את לא טובה בכלום" קיימת הקטנה, במשפט: "אם את ממשיכה לדבר ככה אני ארצח אותך" קיימת אלימות פיזית ונחצה קו אדום, יש לפנות מיד לסיוע של גורמים מוסמכים נוספים.']
    ]
    general_train_object = create_few_shot_train_general_object(general_examples)

    context = [
        'התפקיד שלך כמודל בינה מלאכותית הוא לנתח את השיחה ולציין האם קיימים סוגי אלימות או סכנות מסוימות העולות מתוך השיחה',
        'השיחה מתנהלת בין שני אנשים שלאחד האנשיים קיים חדש שהאיש השני עלול לפגוע בו',
        'אני מגדיר תמרורי אזהרה כסימנים למערכת יחסים לא תקינה- אלו האם תמרורי האזהרה שאני רוצה שתכיר: אובסיביות: מחטט בטלפון הנייד ורוצה לדעת כל דבר לגבי הפרטנר, זוגיות דו פרצופית: בפומבי הפרטנר מציג את עצמו באופן שונה מאשר ביחידים, הקטנה: הפרטנר מפיל על הצד השני את כל האשמה, רגישות קיצונית: הפרטנר מגיב בצורה מאיימת במידה והצד השני מציע להיפרד, הקדוש המעונה: הפרטנר הוא גם התוקף אך מציג את עצמו בתור הקורבן',
        'אני רוצה שבפלט תתייחס לשלושה דברים, 1. אילו סוגי אלימות קיימים בשיחה ובאילו משפטים הם באו לידי ביטוי, 2. אילו תמרורי אזהרה קיימים בשיחה ובאילו משפטים באו לידי ביטוי, 3. קווים אדומים שנחצו וסכנה ממשית שיש להרים לגביה דגל אדום מיד']

    # Define how the model (gpt4) should answer to the question
    system_promt_object = create_context_object(context)

    # Few shot Learning, examples

    conversation = """.אופק, מה אתה חושב שאתה עושה? תראה לי את הטלפון שלך מיד אני רוצה לראות עם מי אתה מדבר,
      אתה יודע שבן אדם אחר היה רוצח אותך, עוד פעם אחת שזה קורה ואתה לא תראה ממני שקל אתה תעבור לגור ברחוב
    """
    conversation_object = create_conversation_object(conversation, 'כעס')

    train_objects = [violence_train_object, warning_train_object, general_train_object]
    promt = get_promt(system_promt_object, train_objects, conversation_object)

    ans = analyse(promt)
    print(ans)