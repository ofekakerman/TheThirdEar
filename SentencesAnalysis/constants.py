DEFAULT_VIOLENCE_EXAMPLES = [
    ['"עוד מילה אחת שיוצאת לך מהפה אני לא קונה לך יותר אוכל"', 'אלימות כלכלית'],
    ['"את לא מתביישת לבקש ממני לדבר אלייך בכבוד אחרי שקניתי לך את הטבעת?"', 'אלימות כלכלית'],
    ['"אני מאוכזב שקנית את התכשיט הזה ב200 שקל, את יודעת שאין לנו את הכסף הזה למותרות"', 'לא זוהתה אלימות בשיחה']

]

DEFAULT_WARNING_SIGN_EXAMPLES = [
    ['"מה קרה שהגעת רק עכשיו הביתה? תגידי לי מיד איפה היית!"', 'אובססיביות'],
    ['"את לא יודעת לעשות שום דבר טוב, לא לבשל, לא כלום"', 'הקטנה']
]

DEFAULT_GENERAL_EXAMPLES = [
    ['"מה קרה? איפה היית אתמול? אני קצת מתבאס שלא ראיתי אותך כל היום"',
     'השיחה תקינה, לא עלו תמרורי אזהרה או סוגי אלימות מובהקים'],
    ['"תשמעי לי טוב, את פשוט אישה רעה את לא טובה בשום דבר אם את ממשיכה להתנהג ככה אני ארצח אותך "',
     'במשפט "את פשוט אישה רעה, את לא טובה בכלום" קיימת הקטנה, במשפט: "אם את ממשיכה לדבר ככה אני ארצח אותך" קיימת אלימות פיזית ונחצה קו אדום, יש לפנות מיד לסיוע של גורמים מוסמכים נוספים.']
]

DEFAULT_CONTEXT = [
    'התפקיד שלך כמודל בינה מלאכותית הוא לנתח את השיחה ולציין האם קיימים סוגי אלימות או סכנות מסוימות העולות מתוך השיחה. בבקשה ציין על האינדקציות אם הן מובהקת או גבוליות לדעתך',
    'השיחה מתנהלת בין שני אנשים שלאחד האנשיים קיים חדש שהאיש השני עלול לפגוע בו',
    'אני מגדיר תמרורי אזהרה כסימנים למערכת יחסים לא תקינה- אלו האם תמרורי האזהרה שאני רוצה שתכיר: אובסיביות: מחטט בטלפון הנייד ורוצה לדעת כל דבר לגבי הפרטנר, זוגיות דו פרצופית: בפומבי הפרטנר מציג את עצמו באופן שונה מאשר ביחידים, הקטנה: הפרטנר מפיל על הצד השני את כל האשמה, רגישות קיצונית: הפרטנר מגיב בצורה מאיימת במידה והצד השני מציע להיפרד, הקדוש המעונה: הפרטנר הוא גם התוקף אך מציג את עצמו בתור הקורבן',
    'אני רוצה שבפלט תתייחס לשלושה דברים, 1. אילו סוגי אלימות קיימים בשיחה ובאילו משפטים הם באו לידי ביטוי, 2. אילו תמרורי אזהרה קיימים בשיחה ובאילו משפטים באו לידי ביטוי, 3. קווים אדומים שנחצו וסכנה ממשית שיש להרים לגביה דגל אדום מיד']

DEFAULT_EMOTION = 'כעס'