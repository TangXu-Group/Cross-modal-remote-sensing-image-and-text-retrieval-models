import torch
import numpy as np


if __name__ == '__main__':
    anns = [
        {'image': 'I1', 'caption': ['T1', 'T2', 'T3', 'T4', 'T5']},
        {'image': 'I2', 'caption': ['T6', 'T5', 'T7', 'T4', 'T8']},
        {'image': 'I3', 'caption': ['T9', 'T10', 'T1', 'T11', 'T7']},
        {'image': 'I4', 'caption': ['T12', 'T13', 'T14', 'T15', 'T16']},
        {'image': 'I5', 'caption': ['T10', 'T17', 'T18', 'T9', 'T19']},
        {'image': 'I6', 'caption': ['T20', 'T21', 'T14', 'T22', 'T23']},
    ]

    img_num = 6
    txt_num = 30
    scores = torch.rand((img_num, txt_num)).cpu().numpy()

    print('score: ')
    for i in scores:
        print(list(i))
    print('------------------------------------------------')
    print('rank score:')
    for i in scores:
        print(list(np.argsort(i)[::-1]))

    # case 1:
    txt2id_dict = {}  # dictionary mapping text to text id
    id2txt_dict = {}  # dictionary mapping text id to text
    text = []         # list stone all text
    image = []        # list stone all image
    txt2img = {}      # dictionary mapping text id to image id
    img2txt = {}      # dictionary mapping image id to text id

    txt_o2img = {}    # dictionary mapping text to image id

    txt_id = -1
    for img_id, ann in enumerate(anns):
        image.append(ann['image'])
        img2txt[img_id] = []
        for i, caption in enumerate(ann['caption']):
            cur_text = caption
            cur_txt_id = txt2id_dict.get(cur_text, None)
            text.append(cur_text)
            if cur_txt_id is None:
                txt_id += 1
                cur_txt_id = txt_id
                txt2id_dict[cur_text] = cur_txt_id
                id2txt_dict[cur_txt_id] = len(text) - 1
                txt2img[cur_txt_id] = []
                txt_o2img[text[id2txt_dict[cur_txt_id]]] = []
            img2txt[img_id].append(cur_txt_id)
            txt2img[cur_txt_id].append(img_id)
            txt_o2img[text[id2txt_dict[cur_txt_id]]].append(img_id)

    # 规范化score矩阵，即相同的图文对，对应的分数应相同
    txt_set = txt2img.keys()
    for txt in range(len(txt_set)):
        iid1 = txt2img[0][0]
        pos1 = iid1 * 5
        for i in range(5):
            if img2txt[iid1][i] == txt:
                break
            pos1 += 1
        for i in range(1, len(txt2img[txt])):
            pos2 = txt2img[txt][i] * 5
            for j in range(5):
                if img2txt[txt2img[txt][i]][j] == txt:
                    break
                pos2 += 1
            scores[iid1, pos1] = scores[iid1, pos2]
            scores[txt2img[txt][i], pos1] = scores[txt2img[txt][i], pos2]

    print('------------------------------------------------')
    print('alter:')
    print('score: ')
    for i in scores:
        print(list(i))
    print('------------------------------------------------')
    print('rank score:')
    for i in scores:
        print(list(np.argsort(i)[::-1]))
    print('------------------------------------------------')
    print('rank score T:')
    print([0, 1, 2, 3, 4, 5])
    print()
    for i in range(len(scores.T)):
        print(list(np.argsort(scores.T[i])[::-1]), i)

    # case 1: alter evaluation
    ranks = np.zeros(scores.shape[0])
    for index, score in enumerate(scores):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            pos = id2txt_dict[i]
            tmp = np.where(inds == pos)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    score_t = scores.T
    ranks = np.zeros(score_t.shape[0])

    for index, score in enumerate(score_t):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in txt2img[txt2id_dict[text[index]]]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # for index, score in enumerate(scores_t2i):
    #     inds = np.argsort(score)[::-1]
    #     ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    print("=================================================================")
    print("case 1: alter")
    print('txt2id_dict: ', txt2id_dict)
    print('id2txt_dict: ', id2txt_dict)
    print('text: ', text)
    print('image: ', image)
    print('txt2img: ', txt2img)
    print('txt_origin2img: ', txt_o2img)
    print('img2txt: ', img2txt)
    print('(tr1, tr5, tr10):', (tr1, tr5, tr10))
    print('(ir1, ir5, ir10):', (ir1, ir5, ir10))
    print("=================================================================")

    # case 2 orgin evaluate:
    text = []
    image = []
    txt2img = {}
    img2txt = {}

    txt_id = 0
    for img_id, ann in enumerate(anns):
        image.append(ann['image'])
        img2txt[img_id] = []
        for i, caption in enumerate(ann['caption']):
            text.append(caption)
            img2txt[img_id].append(txt_id)
            txt2img[txt_id] = img_id
            txt_id += 1

    ranks = np.zeros(scores.shape[0])
    for index, score in enumerate(scores):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    scores_t = scores.T
    ranks = np.zeros(scores_t.shape[0])

    for index, score in enumerate(scores_t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    print("=================================================================")
    print("case 2: orgin")
    print('text: ', text)
    print('image: ', image)
    print('txt2img: ', txt2img)
    print('img2txt: ', img2txt)
    print('(tr1, tr5, tr10):', (tr1, tr5, tr10))
    print('(ir1, ir5, ir10):', (ir1, ir5, ir10))
    print("=================================================================")



