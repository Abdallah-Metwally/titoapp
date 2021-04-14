from __future__ import division
from __future__ import print_function
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg
from spellcheck import SpellCheck
import PreProccessing
import os
from firebase import Firebase, Database
from num import NUMDECODE
import shutil
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


class FilePaths:
    """ filenames and paths to data """
    fnCharList = 'model/charList.txt'
    fnAccuracy = 'model/accuracy.txt'
    fnTrain = 'data/'
    fnCorpus = 'data/corpus.txt'


def train(model, loader):
    """train NN"""
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best validation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occurred
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    """validate NN"""
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation1997 result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate


def segment(file_path):
    """reads images from data/ and outputs the word-segmentation to out/"""

    shutil.rmtree('out/toNN.png')
    # read image, prepare it by resizing it to fixed height and converting it to grayscale
    img = prepareImg(cv2.imread(file_path))

    # execute segmentation with given parameters
    # -kernelSize: size of filter kernel (odd integer)
    # -sigma: standard deviation of Gaussian function used for filter kernel
    # -theta: approximated width/height ratio of words, filter function is distorted by this factor
    # - minArea: ignore word candidates smaller than specified area
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=350)

    # write output to 'out/inputFileName' directory
    if not os.path.exists('out/toNN.png'):
        os.mkdir('out/toNN.png')

    # iterate over all segmented words
    # print('Segmented into %d words' % len(res))
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite('out/toNN.png/%d.png' % j, wordImg)  # save word
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image

    # output summary image with bounding boxes around words
    # cv2.imwrite('out/toNN.png/summary.png', img)


def infer(model, fnImg):
    """recognize text in image provided by file path"""
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    # print('Probability:', '"' + str(round(float(probability[0]) * 100, 2)) + '%"')

    return recognized[0]


def configFirbase():
    # Connect to Firebase
    config = {
        "apiKey": "AIzaSyDKECMZtsUs4WQ5WuztPZMwSCkeN6N5jmM",
        "authDomain": "pharmapp-51930.firebaseapp.com",
        "databaseURL": "https://pharmapp-51930.firebaseio.com",
        "projectId": "pharmapp-51930",
        "storageBucket": "pharmapp-51930.appspot.com",
        "messagingSenderId": "1072958110337",
        "appId": "1:1072958110337:web:621a53ad49fa22fca96b80",
        "measurementId": "G-EEZ1SM3E46",
        "serviceAccount": 'Test Source/pharmapp-51930-firebase-adminsdk-ynf3s-d5cb9b4d74.json'
    }

    firebase = Firebase(config)

    storage = firebase.storage()

    print("Connecting to Firebase....")
    storage.child("ChatImages/1604353532855.jpg").download("Test Source/downloaded.jpg")
    return firebase


def classify(model):
    A = [["", ""], ["", ""], ["", ""], ["", ""], ["", ""]]
    c = 0
    imgFiles = os.listdir('out/toNN.png')
    for (i, f) in enumerate(sorted(imgFiles)):
        res = infer(model, 'out/toNN.png/%s' % f)
        num = NUMDECODE('out/toNN.png/%s' % f)
        A[i][0] = res
        A[i][1] = num
        c = i
        # os.remove('out/toNN.png/%s' % f)
    A = A[:c + 1]
    result = ""
    for i in range(c + 1):
        result += A[i][0] + " "
    spell = SpellCheck("Drugs List/DrugsList.txt", "Drugs List/Dictionary.txt")
    spell.check(result[:-1])
    Final = spell.correct()
    if Final == "":
        # print(result[:-1] + " Cannot be Found")
        result = ""
        for i in range(c + 1):
            if i == c:
                result += A[i][1]
            else:
                result += A[i][0] + " "
        spell.check(result)
        Final = spell.correct()
    if Final == "":
        # print(result + " Cannot be Found")
        result = ""
        for i in range(c + 1):
            if i == c:
                result += " " + A[i][1]
            else:
                result += A[i][0]
        spell.check(result)
        Final = spell.correct()
    if Final == "":
        # print(result + " Cannot be Found")
        result = ""
        result += A[0][0]
        Final = spell.get(result)

    return Final, result


def test(x):
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    true_drug = ["Mucotec 150", "Notussil", "Klacid 500", "Megalase", "Adwiflam", "Comfort", "Omehealth", "Antopral 40",
                 "Motinorm", "Visceralgine", "Buscopan", "Napizole 20", "E-Mox 500", "Levoxin 500", "Picolax",
                 "Spasmo-digestin", "Nexium 40", "Controloc 40", "Spascolon 100", "Fluxopride", "Physiomer", "Cetal",
                 "Simethicone", "Optipred", "Dexatobrin", "Phenadone", "Paracetamol", "Levohistam", "Novactam 750",
                 "Epidron", "Clavimox 457", "Dolo-d", "Megafen-n", "Telfast 120", "Zisrocin 500", "Protozole",
                 "Betadine", "Daktacort", "Gynozol 400", "Lornoxicam", "Dantrelax 50", "Downoprazol 40", "Augmentin",
                 "Alphintern", "Arthrofast 150", "Megamox 457", "Maxilase", "Catafly", "Vitacid C", "Cerebromap",
                 "Escita 10"]
    print("Validating Neural Network")
    imgFiles = os.listdir('D:\Projects\drugs\dataset')
    print('Ground truth -> Recognized')
    for (i, f) in enumerate(sorted(imgFiles)):
        segment(f)
        found, recognized = classify(x, f)
        if found == "":
            recog = infer(x, 'D:/Projects/drugs/dataset/%s' % f)
            spell = SpellCheck("Drugs List/DrugsList.txt", "Drugs List/Dictionary.txt")
            spell.check(recog)
            found = spell.correct()
        true = true_drug[i]
        numWordOK += 1 if true == found else 0
        numWordTotal += 1
        dist = editdistance.eval(found, true)
        numCharErr += dist
        numCharTotal += len(true)
        print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + true + '"', '->',
              '"' + found + '"')

    charAccuracyRate = (numCharTotal - numCharErr) / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character Accuracy: %f%%. Word Accuracy: %f%%.' % (charAccuracyRate * 100.0, wordAccuracy * 100.0))
    return charAccuracyRate, wordAccuracy


def SearchFirebase(firebase, name):
    db = firebase.database()
    db.child("Results").child("medicineName").remove()
    db.child("Results").child("medicineName").set(name)





"""main function"""
def main(file_path):
    model = Model(open(FilePaths.fnCharList).read(), decoderType=DecoderType.BestPath, mustRestore=True)
    # f = configFirbase()
    # PreProccessing.main(firebase=0)
    segment(file_path)
    drug = classify(model)
    # SearchFirebase(f, n)
    #print(open(FilePaths.fnAccuracy).read())
    #model = Model(open(FilePaths.fnCharList).read(), decoderType=DecoderType.BestPath, mustRestore=True)
    #test(model)
    return drug

if __name__ == '__main__':
    main()
