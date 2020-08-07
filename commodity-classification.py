import pandas as pd


def read_file():
    """
    导入数据
    """
    commodity = pd.read_csv(r"D:\PycharmProjects\Commodity-classification\data\train.tsv", sep='\t')
    # 查看分类种类
    commodity['TYPE'].unique()
    print(len(commodity['TYPE'].value_counts()))

    return commodity


if __name__ == '__main__':
    read_file()





from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence = "A Titan RTX has 24GB of VRAM"
# 标记（mask，词汇表当中没有的单词将会被标记出来，拆开后前面还会加上##用来表示这不是单独的单词而是单词中的一部分）
tokenizer_sequence = tokenizer.tokenize(sequence)
# 编码
encoded_sequence = tokenizer(sequence)['input_ids']
# 解码（这是BertModel期望输入的方式）
decoded_sequence = tokenizer.decode(encoded_sequence)


sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."
encoded_sequence_a = tokenizer(sequence_a)['input_ids']
encoded_sequence_b = tokenizer(sequence_b)['input_ids']
print("编码后二者的长度：", len(encoded_sequence_a), " ", len(encoded_sequence_b))
"""
由于二者的长度不相同，不能按照原先的长度放入张量中，因此需要将第一个序列填充到第二个的长度，或者将第二个序列截取到第一个序列的长度。
在第一种情况下，ID列表由填充索引扩展。我们可以将列表传递给令牌生成器，并要求其像这样填充：
"""
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
print("查看二者序列长度：（此时两者序列长度填充为相同）\n", padded_sequences)
print(padded_sequences.keys())
# 掩码是二进制张量，指示填充索引的位置，以便模型不会注意它们
print(padded_sequences['attention_mask'])

"""
我们可以使用tokenizer生成器通过将两个序列作为两个参数（而不是像以前的列表）传递自动生成这样的句子
"""
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"
encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict['input_ids'])
print(decoded)
# 对于某些模型而言，这足以了解一个序列在何处结束以及另一序列在何处开始。
# 但是，其他模型（例如BERT）具有其他机制，即token type IDs（also called segment IDS）。它们是识别模型中不同序列的二进制掩码。
# 第一个序列，即用于问题的“上下文”，其所有标记均由表示0，而问题的所有标记均由表示1,。某些模型（例如）XLNetModel使用以表示的附加token用2。
print(encoded_dict['token_type_ids'])

from transformers import pipeline

# 情感分析
nlp = pipeline("sentiment-analysis")
result = nlp("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
result = nlp("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

"""
两个序列之间的联系
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ['not paraphrase', 'is paraphrase']
sequence_0 = "The company HuggingFace is based in New Work City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")  # pt代表使用PyTorch
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase)[0]  # 元组取索引0元素，tensor([[-0.3418, 1.8805]], grad_fn=<AddmmBackward>)
not_paraphrase_classification_logits = model(**not_paraphrase)[0]

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")  # paraphrase_results[i] * 100，代表重复100次
# 结果：
# not paraphrase: 10%
# is paraphrase: 90%

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
# 结果
# not paraphrase: 94%
# is paraphrase: 6%



"""
提取式回答run_squad.py
"""
from transformers import pipeline

nlp = pipeline("question-answering")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a 
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune 
a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
"""

result = nlp(questin='What is extraction answering?', context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

result = nlp(questin='What is a good example of a question answering dataset?', context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")


"""
问答系统
"""
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
