# Chain of Thought
# Question -> LM -> Tought + Answer

import dspy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key="API-KEY-HERE")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="often between 1 and 5 words")

question = "Who provided the assist for goal in football world cup finals in 2014?"

generate_answer = dspy.ChainOfThought(QA)
prediction = generate_answer(question=question)

#print(prediction.rationale)
print(prediction.answer)

lm.inspect_history(1)

multi_step_question = "What is the capital of the birth state of the person who provided the assist for the Mario Gotze's goal in football world cup finals in 2014?"
output = generate_answer(question=multi_step_question)
print(output)

### DSPy Modules

import dspy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key="AIzaSyA3jdWQ7LxYS1I8FaMyb5wOZl796J_rntY")
dspy.configure(lm=lm)

class QA(dspy.Signature):
    """Given the question, generate the answer"""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class ChainOfThoughtModule(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(QA)
    
    def forward(self, question):
        return self.cot(question=question)

multi_step_question2 = "What is the capital of the birth state of the person who provided the assist for the Mario Gotze's in football world cup finals in 2014?"

cot = ChainOfThoughtModule()
output = cot(question=multi_step_question2)
print(output)

### We can stack multiple LLM Calls inside our Models

class DoubleChainOfThought(dspy.Module):
    def __init__(self):
        self.cot1 = dspy.ChainOfThought("question -> step_by_step_thought") # Sadece soruya bakarak düşünme üretiyor. (Giriş -> Question, Çıkış -> Step by Step Thought)
        self.cot2 = dspy.ChainOfThought("question, thought -> one_word_answer") # Hem soruya hem de düşünceye bakarak cevap üretiyor. 

    def forward(self, question):
        thought = self.cot1(question=question).step_by_step_thought # Soruyu alıyor, reasoning açıklaması döndürüyor.
        answer = self.cot2(question=question, thought=thought).one_word_answer # Reasoning'e ve soruya bakarak tek kelimelik cevabı çıkarıyor.
        return dspy.Prediction(thought=thought, answer=answer)
    

multi_step_question3 = "What is the capital of the birth state of the person who provided the assist for the Mario Gotze's in football world cup finals in 2014?"

doubleCot = DoubleChainOfThought()
output = doubleCot(question=multi_step_question3)
print(output)

