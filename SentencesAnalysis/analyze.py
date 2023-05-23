import openai


def analyse(promt):
    completion = openai.ChatCompletion.create(
      model='gpt-4',
      temperature=0.4,
      messages=promt
    )
    # Get our answer
    answer = completion.choices[0].message.content
    return answer
