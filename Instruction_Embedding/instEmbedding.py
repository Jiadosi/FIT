from gensim.models import word2vec, KeyedVectors
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
# from spicy.spatial import distance

# preparing dataset
def preparing(dirPath):
    print('---preparing dataset---')
    for f in os.listdir(dirPath):
        filePath = ''
        if 'mips' in f:
            filePath = os.path.join(dirPath, f)
        if not filePath:
            continue
        print(filePath)
        with open(filePath, 'r') as f:
            data = f.readlines()
        res = []
        with open('./w2v_mips_dataset.txt', 'a') as f:
            for line in data:
                g = json.loads(line)
                for bb in g['features']:
                    f.write(' '.join(bb[-1]))
                    f.write('\n')
                    res.append(' '.join(bb[-1]))

# preparing input
def inputGen(filePath):
    sentences = word2vec.LineSentence(filePath)
    return sentences

# training model, sg=1:skip-gram, hs=0:negative sampling
def training(modelPath, sentences):
    print('---training model---')
    model = word2vec.Word2Vec(sg=1, size=100, window=5, min_count=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save(modelPath)

# loading model
def loading(modelPath):
    #model = KeyedVectors.load_word2vec_format()
    model = word2vec.Word2Vec.load(modelPath)
    return model

# visualizing

def display_allwords(model1, model2):
    # model1
    vocab1 = list(model1.wv.vocab)
    vocab1 = vocab1[:3000]
    X = model1[vocab1]

    # model2
    vocab2 = list(model2.wv.vocab)
    vocab2 = vocab2[:3000]
    Y = model2[vocab2]
    Z = np.concatenate((X, Y))

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(Z)
    df = pd.DataFrame(X_tsne, index=vocab1+vocab2, columns=['x', 'y'])
    plt.scatter(df['x'], df['y'], s=10, c='m')
    # for word, pos in df.iterrows():
    #     plt.annotate(word, pos, alpha=0.3, size=10)
    plt.savefig('./x86_arm.png', format='png')
    
def display_closestwords_tsnescatterplot(model, word, size):

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.most_similar(word, topn = 20)
    '''
    # trick
    tmp = []
    for _ in close_words:
        if 'mov' in _[0]:
            tmp.append(_)
    close_words = tmp
    '''
    print(close_words)

    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        # print(wrd_score)
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0, perplexity=1)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+1, x_coords.max()+500)
    plt.ylim(y_coords.min()+1, y_coords.max()+500)

    plt.show()
    # plt.savefig('./x86_sub.png', format='png')



if __name__ == "__main__":
    modelPath = './myModel/x86'
    modelPath2 = './myModel/arm'
    dirPath = './filtered_json_inst'
    filePath = './w2v_arm_dataset.txt'
    # preparing(dirPath)
    # sentences = inputGen(filePath)
    # training(modelPath, sentences)
    model = loading(modelPath)
    model2 = loading(modelPath2)
    #model = loading(modelPath2)
    #print(model['mov~eax,<TAG>'])
    #print(model['mov~eax,<eax+0>'])

    xtest = model['test~eax,eax']
    xmov = model['mov~eax,0']
    xadd = model['add~eax,0']
    acmp = model2['CMP~R0,0']
    aldr = model2['LDR~R3,[R5+0]']
    aadd = model2['ADD~SP,SP,0']
    dis1 = np.dot(xtest, acmp)/(np.linalg.norm(xtest)*(np.linalg.norm(acmp)))
    print(dis1)
    dis1 = np.dot(xmov, aldr)/(np.linalg.norm(xmov)*(np.linalg.norm(aldr)))
    print(dis1)
    # mov~eax,<TAG>  movsx~ebx,[esp+0]
    # display_closestwords_tsnescatterplot(model2, 'LDR~R3,[R5+0]', 100)
    # display_allwords(model1, model2)
