# import spacy
# import torch
# import torch.nn.functional as F
#
#
# if __name__ == '__main__':
#     # nlp = spacy.load("en_core_web_sm")
#     # text = 'There are two house with a blue roof by the side of the road.'
#     # doc = nlp(text)
#     #
#     # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
#     # print("Verbs:", [(token.lemma_, token.i) for token in doc if token.pos_ == "VERB"])
#     # print("Nouns:", [(token.lemma_, token.i) for token in doc if token.pos_ == "NOUN"])
#     # print("Nums:", [(token.lemma_, token.i) for token in doc if token.pos_ == "NUM"])
#     # print("Dets:", [(token.lemma_, token.i) for token in doc if token.pos_ == "DET"])
#
#     bs = 64
#
#     prob = torch.tensor([[0.41, 0.59] for i in range(bs)])
#     prob.requires_grad = True
#     label1 = torch.tensor([[1] for i in range(bs)])
#     label2 = torch.cat([1 - label1, label1], dim=1)
#
#     loss1 = F.cross_entropy(prob, label1.squeeze(1))
#
#     threshold = 1.18e-38
#     entropy = -F.log_softmax(prob, dim=1) * label2
#     entropy_grad = entropy[entropy >= threshold]
#     entropy_ng = entropy[entropy < threshold].detach()
#
#     loss2 = torch.sum(entropy_grad) / entropy.shape[0]
#
#     loss3 = F.nll_loss(F.log_softmax(prob, dim=1), label1.squeeze(1))
#
#     loss4 = -torch.sum(F.log_softmax(prob, dim=1) * label2, dim=1).mean()
#
#     loss5 = -F.log_softmax(prob, dim=1)[:, 1].mean()
#     loss6 = -(F.log_softmax(prob, dim=1)[:, 1] * 1.0).mean()
#
#     print('loss1: ', float(loss1))
#     print('loss2: ', float(loss2))
#     print('loss3: ', float(loss3))
#     print('loss4: ', float(loss4))
#     print('loss5: ', float(loss5))
#     print('loss6: ', float(loss6))
#     print('loss1 == loss2: ', float(loss1) == float(loss2))
#     print('loss1 == loss3: ', float(loss1) == float(loss3))


#
# import requests
# import torch
# from PIL import Image
# from transformers import AlignProcessor, AlignModel
#
# processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
# model = AlignModel.from_pretrained("kakaobrain/align-base")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# candidate_labels = ["an image of a cat", "an image of a dog"]
#
# inputs = processor(text=candidate_labels, images=image, return_tensors="pt")
#
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # this is the image-text similarity score
# logits_per_image = outputs.logits_per_image
# # we can take the softmax to get the label probabilities
# probs = logits_per_image.softmax(dim=1)
# print(probs)


from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch


# if __name__ == '__main__':
#     tokenizer = BertTokenizer.from_pretrained('/data1/amax/pretrain_model/bert-base-uncased')
#     model = BertModel.from_pretrained("/data1/amax/pretrain_model/bert-base-uncased")
#     text1 = "this port is crammed with ships docking in here."
#     text2 = "Near the beach residential area there is a port moored in a row."
#     text3 = "There is a harbor near a residential area by the sea, and the boats are moored side by side"
#     text4 = "There is a ship sailing on the sea"
#
#     encoded_input1 = tokenizer(text1, return_tensors='pt')
#     encoded_input2 = tokenizer(text2, return_tensors='pt')
#     encoded_input3 = tokenizer(text3, return_tensors='pt')
#     encoded_input4 = tokenizer(text4, return_tensors='pt')
#
#     output1 = model(**encoded_input1).last_hidden_state[0][0].unsqueeze(0)
#     output2 = model(**encoded_input2).last_hidden_state[0][0].unsqueeze(0)
#     output3 = model(**encoded_input3).last_hidden_state[0][0].unsqueeze(0)
#     output4 = model(**encoded_input4).last_hidden_state[0][0].unsqueeze(0)
#
#     s1 = F.cosine_similarity(output1, output2)
#     s2 = F.cosine_similarity(output1, output3)
#     s3 = F.cosine_similarity(output2, output3)
#     s4 = F.cosine_similarity(output4, output3)
#
#     sim = torch.cat([s1, s3], dim=-1)
#     sim = sim - torch.min(sim)
#     sim /= torch.max(sim) - torch.min(sim)
#     sim = torch.softmax(sim, dim=-1)
#
#     print(s1, s2, s3, s4)
#     print(sim)

import torch
import re


if __name__ == '__main__':
    txt = 'text_encoder.bert.encoder.layer.11.output.LayerNorm.weight'
    matchStr = '.+\..+\..+\..+\.(\d+)\..*'
    match = re.match(matchStr, txt)
    print(match.group(0))
    print(match.group(1))