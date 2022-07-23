# %%
import glob
import pandas as pd
import numpy as np
# %% declare variables

D = 10000  # Dimensions of HyperVector
N = 3  # N-gram
# %% 成分のうち，半分は1でもう半分は-1のハイパーベクトルを生成する．


def genRandomHV(D):
    if (D % 2):
        print('Dimension is odd!!')
    else:
        randomIndex = np.random.permutation(D)
        randomHV = np.zeros(D)

        for i in range(0, round(D/2)):
            randomHV[randomIndex[i]] = 1
        for i in range(round(D/2), D):
            randomHV[randomIndex[i]] = -1

    return randomHV

# %% 各文字に対応するhyperVectorを格納するItemMemoryを参照する．．


def lookupItemMemory(itemMemory, key, D):
    if(itemMemory.filter(items=[key]).empty):
        tmp = pd.Series(data=[genRandomHV(D)], index=[key])
        itemMemory = pd.concat([itemMemory, tmp])
        randomHV = itemMemory[key]
    else:
        randomHV = itemMemory[key]
    return itemMemory, randomHV

# %% 2つのベクトルのコサイン類似度を求める．


def cosAngle(u, v):
    cos = np.dot(u, v)/(np.linalg.norm(u, ord=2)*np.linalg.norm(v, ord=2))
    return cos

# %% 文章をハイパーベクトルにする．N-gramの文字数分のhyperVectorをcombineして足し合わせていく．


def computeSumHV(buffer, itemMemory, N, D):
    block = np.zeros((N, D))
    sumHV = np.zeros(D)

    for numItems in range(0, len(buffer), 1):
        key = buffer[numItems]

        block = np.roll(block, (1, 1), axis=(0, 1))
        [itemMemory, block[0]] = lookupItemMemory(itemMemory, key, D)

        if (numItems >= (N - 1)):
            nGrams = block[0]
            for i in range(1, N, 1):
                nGrams = np.multiply(nGrams, block[i])
            sumHV = sumHV + nGrams

    return itemMemory, sumHV

# %% hyperVectorをbinarizeする


def binarizeHV(v):
    threshold = 0
    for i in range(0, len(v), 1):
        if (v[i] > threshold):
            v[i] = 1
        else:
            v[i] = -1

    return v

# %%


def binarizeLanguageHV(langAM):
    langLabels = [
        'afr',
        'bul',
        'ces',
        'dan',
        'nld',
        'deu',
        'eng',
        'est',
        'fin',
        'fra',
        'ell',
        'hun',
        'ita',
        'lav',
        'lit',
        'pol',
        'por',
        'ron',
        'slk',
        'slv',
        'spa',
        'swe'
    ]

    for i in range(0, len(langLabels), 1):
        v = langAM[langLabels[i]]
        langAM[langLabels[i]] = binarizeHV(v)

    return langAM

# %% 訓練データを読み込んで学習する．各言語ごとにcomputeSumHVでハイパーベクトルを生成した後binarizeして，AssociativeMemoryに格納する．


def buildLanguageHV(N, D):
    iM = pd.Series(dtype='object')
    langLabels = [
        'afr',
        'bul',
        'ces',
        'dan',
        'nld',
        'deu',
        'eng',
        'est',
        'fin',
        'fra',
        'ell',
        'hun',
        'ita',
        'lav',
        'lit',
        'pol',
        'por',
        'ron',
        'slk',
        'slv',
        'spa',
        'swe'
    ]
    langAM = pd.Series(index=langLabels, dtype='object')
    for i in range(0, len(langLabels), 1):
        fileAddress = './training_texts/' + langLabels[i] + '.txt'
        fileID = open(fileAddress)
        buffer = fileID.read()
        fileID.close()
        print('Loaded training language file %s' % fileAddress)

        [iM, langHV] = computeSumHV(buffer, iM, N, D)
        langAM[langLabels[i]] = langHV

    return iM, langAM

# %% buildLanguageHVで学習し生成したlangHVと，テストデータから生成したhyperVectorをそれぞれコサイン類似度を用いて比較し，もっとも類似度の大きかった言語でテストデータをラベリングする．


def test(iM, langAM, N, D):
    total = 0
    correct = 0
    predicLang = 'foo'
    langLabels = [
        'afr',
        'bul',
        'ces',
        'dan',
        'nld',
        'deu',
        'eng',
        'est',
        'fin',
        'fra',
        'ell',
        'hun',
        'ita',
        'lav',
        'lit',
        'pol',
        'por',
        'ron',
        'slk',
        'slv',
        'spa',
        'swe'
    ]
    list_test = [
        'af',
        'bg',
        'cs',
        'da',
        'nl',
        'de',
        'en',
        'et',
        'fi',
        'fr',
        'el',
        'hu',
        'it',
        'lv',
        'lt',
        'pl',
        'pt',
        'ro',
        'sk',
        'sl',
        'es',
        'sv'
    ]
    langMap = pd.Series(data=langLabels, index=list_test)
    fileList = sorted(glob.glob('./testing_texts/*.txt'))
    for i in range(0, len(fileList), 1):
        actualLabel = fileList[i]
        actualLabel = actualLabel[16:18]

        fileAddress = fileList[i]
        fileID = open(fileAddress)
        buffer = fileID.read()
        fileID.close()
        print('Loaded testing language file %s' % fileAddress)

        [iMn, textHV] = computeSumHV(buffer, iM, N, D)
        textHV = binarizeHV(textHV)
        if np.all(iM != iMn):
            print('>>>>>   NEW UNSEEN ITEM IN TEST FILE   <<<<<')
            break
        else:
            maxAngle = -1
            for l in range(0, len(langLabels), 1):
                angle = cosAngle(langAM[langLabels[l]], textHV)
                if(angle > maxAngle):
                    maxAngle = angle
                    predicLang = langLabels[l]
            if predicLang == langMap[actualLabel]:
                correct = correct + 1
            else:
                print('%s ---> %s ' % (langMap[actualLabel], predicLang))
            total = total + 1
    accuracy = correct/total

    return accuracy


# %% training


[iM, langAM] = buildLanguageHV(N, D)
langAM = binarizeLanguageHV(langAM)

# %% testing


accuracy = test(iM, langAM, N, D)
print('accuracy')
print(accuracy)
# %%
