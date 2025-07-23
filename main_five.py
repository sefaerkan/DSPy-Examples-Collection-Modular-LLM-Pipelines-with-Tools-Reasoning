# Multi-Hop Reasoning
# Query the database multiple times before arriving at the answer -> Birden fazla sorgu yaparak sonuca ulaşma

import dspy
from dspy.dsp.utils import deduplicate
lm = dspy.LM("gemini/gemini-2.5-flash", api_key="API-KEY-HERE")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

class GenerateQuery(dspy.Signature):
    context = dspy.InputField(desc = "may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc = "may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc = "often between 1 and 5 words")


class MultiHop(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=3):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query_response = self.generate_query[hop](context=context, question=question)
            query = query_response.query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    

question = "What is the date of birth of the player who provided the assist for the final goal in football world cup finals in 2014?"
multi_hop = MultiHop()
output = multi_hop(question=question)
print("Context: \n" )
_ = [print(c) for c in output.context]
print("\nAnswer: ", output.answer)