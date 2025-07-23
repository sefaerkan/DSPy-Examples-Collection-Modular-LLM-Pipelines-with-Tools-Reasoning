# Outputting Typed Predictors

import dspy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key="API-KEY-HERE")
dspy.configure(lm=lm)

from pydantic import BaseModel, Field

class AnswerConfidence(BaseModel):
    answer : str = Field("Answer. 1-5 words")
    confidence : float = Field("Your confidence between 0-1")

class QAWithConfidence(dspy.Signature):
    """Given user's question, answer it and also give your confidence value"""
    question = dspy.InputField()
    answer: AnswerConfidence = dspy.OutputField()

predict = dspy.ChainOfThought(QAWithConfidence)

question = "Who provided the assist for the goal in football world cup finals in 2014?"

output = predict(question=question)
print(output.answer)
print(output.answer.confidence)

## More Complex TypedPredictors

from pydantic import BaseModel, Field

class Answer(BaseModel):
    country: str = Field()
    year : int = Field()

class QAList(dspy.Signature):
    """Given user's question, answer with a JSON readable python list"""
    question = dspy.InputField()
    answer_list: list[Answer] = dspy.OutputField()

question2 = "Generate a list of country and the year of FIFA world cup winners from 2002-present"
predict2 = dspy.ChainOfThought(QAList)
answer = predict2(question=question2)
print(answer.answer_list)