# Retrieval Augmented Generation (RAG)

import dspy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key="API-KEY-HERE")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

retrieve = dspy.Retrieve(k=3)
topK_passages = retrieve("Andre Schürrle").passages

topK_passages

class GenerateAnswer(dspy.Signature):
    """Answer quesitons with short factoid answers"""
    context = dspy.InputField(desc = "may contain relevant facts") # belge içeriği
    question = dspy.InputField()
    answer = dspy.OutputField(desc = "often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
    
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return prediction 
    
question = "What is the date of birth of the player who provided the assist for the final goal in football world cup in 2014?"
rag = RAG()
output = rag(question=question)
print(output)

lm.inspect_history(n=1)