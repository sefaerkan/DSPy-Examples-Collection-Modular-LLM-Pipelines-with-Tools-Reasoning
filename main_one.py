# Programming - Not Prompting - Language Models

import dspy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key="API-KEY-HERE")
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")

prediction = predict(question="Who scored the final goal in football world cup finals in 2014?")
print(prediction.answer)

lm.inspect_history(1)


# Signature - defining a signature for a question-answering task

class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

predict = dspy.Predict(QA) # Instead of dspy.Predict("question -> answer")
prediction = predict(question="Who scored the final goal in football world cup finals in 2014?")
print(prediction.answer)

lm.inspect_history(1)

# Signature (with comment)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

predict = dspy.Predict(QA) # Instead of dspy.Predict("question -> answer")
prediction = predict(question="Who scored the final goal in football world cup finals in 2014?")
print(prediction.answer)

lm.inspect_history(1)