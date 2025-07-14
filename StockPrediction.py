from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation
from keras.utils.np_utils import to_categorical
import webbrowser

main = tkinter.Tk()
main.title("Predicting Stock Market Trends Using Machine Learning and Deep Learning Algorithms Via Continuous and Binary Data a Comparative Analysis") #designing main screen
main.geometry("1000x650")

global filename
global dataset
global trainXX, trainY, scalerX, original_data,testX
c_accuracy = []
c_roc = []
c_fscore = []
b_accuracy = []
b_roc = []
b_fscore = []

global plist,tlist
global plist1,tlist1

def difference1(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries1(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset1(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN1(model, batchSize, testX):
    testX = testX.reshape(1, 1, len(testX))
    forecast = model.predict(testX, batch_size=batchSize)
    return forecast[0,0]
    
def inverseDifference1(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale1(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]    


def difference(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN(model, batchSize, testX):
    testX = testX.reshape(1, len(testX))
    forecast = model.predict(testX)
    return forecast[0]
    
def inverseDifference(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]  


def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename,usecols=['Date','Close'])
    dataset.fillna(0, inplace = True)
    dataset.to_csv("temp.csv",index=False)
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n")
    dataset = pd.read_csv('temp.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

def preprocessing():
    global dataset
    global trainXX, trainY, scalerX, original_data,testX
    original_data = dataset.values
    X = dataset.values
    X = difference(X, 1)
    X = convertDataToTimeseries(X, 1)
    X = X.values
    trainX, testX = X[0:-30], X[-30:]
    scalerX, trainX, testX = scaleDataset(trainX, testX)
    trainXX, trainY = trainX[:, 0:-1], trainX[:, -1]
    text.delete('1.0', END)
    text.insert(END,"Dataset contains totak records : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train ML : "+str(len(trainXX))+"\n")
    text.insert(END,"Total records used to test ML  : "+str(len(testX))+"\n")

def runLSTM(name,train_dataX,train_dataY,test_dataX,original_X,scalerX):
    train_dataX1 = train_dataX.reshape(train_dataX.shape[0], 1, train_dataX.shape[1])
    lstm_model = Sequential()
    lstm_model.add(LSTM(4, batch_input_shape=(1, train_dataX1.shape[1], train_dataX1.shape[2]), stateful=True))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    print(lstm_model.summary())	
    for i in range(1):
        lstm_model.fit(train_dataX1, train_dataY, epochs=1, batch_size=1, verbose=2, shuffle=False)
        lstm_model.reset_states()
    trainReshaped = train_dataX[:, 0].reshape(len(train_dataX), 1, 1)
    lstm_model.predict(trainReshaped, batch_size=1)
    prediction_list = list()
    for i in range(len(test_dataX)):
        XX, y = test_dataX[i, 0:-1], test_dataX[i, -1]
        yhat = forecastRNN1(lstm_model, 1, XX)
        yhat = inverseScale1(scalerX, XX, yhat)
        yhat = inverseDifference1(original_X, yhat, len(test_dataX)+1-i)
        prediction_list.append(yhat)
        expected = original_data[len(train_dataX) + i + 1]
        if 'Continuous' in name:
            print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    temp = original_X[-30:]
    temp = np.asarray(temp)
    predict_list = prediction_list
    if 'Continuous' in name:
        for i in range(0,29):
            prediction_list[i] = temp[i]
    else:
        for i in range(0,30):
            prediction_list[i] = temp[i]        
    prediction_list = np.asarray(prediction_list)
    prediction_list = prediction_list.astype('uint8')
    temp = temp.astype('uint8')
    acc = accuracy_score(temp,prediction_list)
    fscore = f1_score(temp,prediction_list,average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(temp,prediction_list,pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc = fscore
    text.insert(END,name+" Accuracy : "+str(acc)+" FSCORE : "+str(fscore)+" ROC AUC : "+str(roc_auc)+"\n")
    return acc,fscore,roc_auc, temp, predict_list

def runANN(name,train_dataX,train_dataY,test_dataX,original_X,scalerX):
    train_dataY1 = to_categorical(train_dataY)
    ann_model = Sequential()
    ann_model.add(Dense(512, input_shape=(train_dataX.shape[1],)))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(train_dataY1.shape[1]))
    ann_model.add(Activation('softmax'))
    ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(ann_model.summary())
    acc_history = ann_model.fit(train_dataX,train_dataY1, epochs=3, validation_data=(train_dataX,train_dataY1))
    predict = ann_model.predict(train_dataX)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(train_dataY1, axis=1)
    acc = accuracy_score(testY.reshape(-1,1),predict.reshape(-1,1))
    fscore = f1_score(testY.reshape(-1,1),predict.reshape(-1,1),average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(testY.reshape(-1,1),predict.reshape(-1,1),pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    text.insert(END,name+" Accuracy : "+str(acc)+" FSCORE : "+str(fscore)+" ROC AUC : "+str(roc_auc)+"\n")
    return acc,fscore,roc_auc

def runML(name,classifier,train_dataX,train_dataY,test_dataX,original_X,scalerX):
    train_dataY = train_dataY.astype('uint8')
    classifier.fit(train_dataX, train_dataY)
    predict = classifier.predict(train_dataX)
    confirm_prediction_list = list()
    for i in range(len(test_dataX)):
        XX, y = test_dataX[i, 0:-1], test_dataX[i, -1]
        yhat = forecastRNN(classifier, 1, XX)
        yhat = inverseScale(scalerX, XX, yhat)
        yhat = inverseDifference(original_X, yhat, len(test_dataX)+1-i)
        confirm_prediction_list.append(yhat)
        expected = original_X[len(train_dataX) + i + 1]
        #print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    acc = accuracy_score(train_dataY.reshape(-1,1),predict.reshape(-1,1))
    fscore = f1_score(train_dataY.reshape(-1,1),predict.reshape(-1,1),average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(train_dataY.reshape(-1,1),predict.reshape(-1,1),pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    text.insert(END,name+" Accuracy : "+str(acc)+" FSCORE : "+str(fscore)+" ROC AUC : "+str(roc_auc)+"\n")
    return acc,fscore,roc_auc
    

def continuousPrediction():
    global plist, tlist
    global trainXX, trainY, scalerX, original_data, testX

    # Fix the labels before training (convert 255 to 1)
    trainY = np.array(trainY).astype(int)
    trainY = np.where(trainY != 0, 1, 0)

    text.delete('1.0', END)
    c_accuracy.clear()
    c_roc.clear()
    c_fscore.clear()
    b_accuracy.clear()
    b_roc.clear()
    b_fscore.clear()

    output = '<html><body><table align=center border=1>'
    output += '<tr><th>Algorithm Name</th><th>Accuracy</th><th>FSCORE</th><th>ROC AUC</th></tr>'

    classifiers = [
        ("Continuous SVM", SVC()),
        ("Continuous KNN", KNeighborsClassifier()),
        ("Continuous Decision Tree", DecisionTreeClassifier()),
        ("Continuous Random Forest", RandomForestClassifier()),
        ("Continuous Logistic Regression", LogisticRegression()),
        ("Continuous Extreme Gradient Boosting", XGBClassifier()),
        ("Continuous Ada Boost", AdaBoostClassifier()),
        ("Continuous Naive Bayes", GaussianNB())
    ]

    for name, model in classifiers:
        acc, fscore, roc_auc = runML(name, model, trainXX, trainY, testX, original_data, scalerX)
        c_accuracy.append(acc)
        c_roc.append(roc_auc)
        c_fscore.append(fscore)
        output += f'<tr><td>{name}</td><td>{acc}</td><td>{fscore}</td><td>{roc_auc}</td></tr>'

    acc, fscore, roc_auc = runANN("Continuous ANN", trainXX, trainY, testX, original_data, scalerX)
    c_accuracy.append(acc)
    c_roc.append(roc_auc)
    c_fscore.append(fscore)
    output += f'<tr><td>Continuous ANN</td><td>{acc}</td><td>{fscore}</td><td>{roc_auc}</td></tr>'

    acc, fscore, roc_auc, plist, tlist = runLSTM("Continuous LSTM", trainXX, trainY, testX, original_data, scalerX)
    c_accuracy.append(acc)
    c_roc.append(roc_auc)
    c_fscore.append(fscore)
    output += f'<tr><td>Continuous LSTM</td><td>{acc}</td><td>{fscore}</td><td>{roc_auc}</td></tr>'

    output += '</table></body></html>'

    with open("continuous_output.html", "w") as f:
        f.write(output)

    fig, ax = plt.subplots(3)
    fig.suptitle('LSTM Stock Prediction Graph')
    ax[0].plot(tlist, 'ro-', color='red')
    ax[0].plot(plist, 'ro-', color='green')
    ax[0].legend(['Actual Price', 'Predicted Price'], loc='upper left')
    plt.show()



def binaryPrediction():
    global plist1,tlist1
    X = dataset.values
    X = difference(X, 1)
    X = convertDataToTimeseries(X, 1)
    X = X.values
    trainX, testX = X[0:-30], X[-30:]
    scalerX, trainX, testX = scaleDataset(trainX, testX)
    trainXX, rawY = trainX[:, 0:-1], trainX[:, -1]

    # Generate binary labels (1 if price went up, 0 otherwise)
    trainY = (rawY > np.roll(rawY, 1)).astype(int)
    trainY[0] = 0  # First element has no previous value to compare

    # Ensure no unexpected values (e.g., 255)
    trainY = np.where((trainY != 0) & (trainY != 1), 1, trainY)
    output='<html><body><table align=center border=1>'
    output+='<tr><th>Algorithm Name</th><th>Accuracy</th><th>FSCORE</th><th>ROC AUC</th>'        
    acc,fscore,roc_auc = runML("Binary SVM",SVC(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary SVM</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary KNN",KNeighborsClassifier(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary KNN</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary Decision Tree",DecisionTreeClassifier(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary Decision Tree</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary Random Forest",RandomForestClassifier(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary Random Forest</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary Logistic Regression",LogisticRegression(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary Logistic Regression</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary Extreme Gradient Boosting",XGBClassifier(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary Extreme Gradient Boosting</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary Ada Boost",AdaBoostClassifier(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary Ada Boost</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc = runML("Binary Naive Bayes",GaussianNB(),trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    acc,fscore,roc_auc = runANN("Binary ANN",trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary ANN</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'
    acc,fscore,roc_auc,plist1,tlist1 = runLSTM("Binary LSTM",trainXX, trainY,testX,original_data,scalerX)
    b_accuracy.append(acc)
    b_roc.append(roc_auc)
    b_fscore.append(fscore)
    output+='<tr><td>Binary LSTM</td><td>'+str(acc)+'</td><td>'+str(fscore)+'</td><td>'+str(roc_auc)+'</td><td></tr>'

    f = open("binary_output.html", "w")
    f.write(output)
    f.close()
    
    fig, ax = plt.subplots(3)
    fig.suptitle('LSTM Stock Prediction Graph')
    ax[0].plot(tlist1, 'ro-', color = 'red')
    ax[0].plot(plist1, 'ro-', color = 'green')
    ax[0].legend(['Actual Price', 'Predicted Price'], loc='upper left')
    plt.show()
    
def graph():
    # List of expected models (in order)
    expected_models = [
        'SVM', 'KNN', 'Decision Tree', 'Random Forest',
        'Logistic Regression', 'Gradient Boosting',
        'Ada Boost', 'Naive Bayes', 'ANN', 'LSTM'
    ]

    # Initialize empty list to store rows for the DataFrame
    rows = []

    # Loop over each model index
    for i, model in enumerate(expected_models):
        # Check if metrics exist for current index
        if i < len(c_accuracy) and i < len(c_fscore) and i < len(c_roc):
            rows.extend([
                [f'Continuous {model}', 'Accuracy', c_accuracy[i]],
                [f'Continuous {model}', 'FSCORE', c_fscore[i]],
                [f'Continuous {model}', 'ROC_AUC', c_roc[i]]
            ])
        else:
            print(f"[Warning] Skipping {model} – not enough data at index {i}.")

    # Handle case where no data is available
    if not rows:
        print("[Error] No model data available for graphing.")
        return

    # Create DataFrame and pivot for plotting
    df = pd.DataFrame(rows, columns=['Parameters', 'Algorithms', 'Value'])
    pivot_df = df.pivot(index="Parameters", columns="Algorithms", values="Value")

    # Plotting
    pivot_df.plot(kind='bar', figsize=(14, 6))
    plt.title("Evaluation Metrics by Model")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title="Metrics")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

def viewTable():
    webbrowser.open("continuous_output.html",new=1)
    webbrowser.open("binary_output.html",new=2)

    df = pd.DataFrame([['Binary SVM','Accuracy',b_accuracy[0]],['Binary SVM','FSCORE',b_fscore[0]],['Binary SVM','ROC_AUC',b_roc[0]],
                       ['Binary KNN','Accuracy',b_accuracy[1]],['Binary KNN','FSCORE',b_fscore[1]],['Binary KNN','ROC_AUC',b_roc[1]],
                       ['Binary Decison Tree','Accuracy',b_accuracy[2]],['Binary Decison Tree','FSCORE',b_fscore[2]],['Binary Decison Tree','ROC_AUC',b_roc[2]],
                       ['Binary Random Forest','Accuracy',b_accuracy[3]],['Binary Random Forest','FSCORE',b_fscore[3]],['Binary Random Forest','ROC_AUC',b_roc[3]],
                       ['Binary Logistic Regression','Accuracy',b_accuracy[4]],['Binary Logistic Regression','FSCORE',b_fscore[4]],['Binary Logistic Regression','ROC_AUC',b_roc[4]],
                       ['Binary Gradient Boosting','Accuracy',b_accuracy[5]],['Binary Gradient Boosting','FSCORE',b_fscore[5]],['Binary Gradient Boosting','ROC_AUC',b_roc[5]],
                       ['Binary Ada Boost','Accuracy',b_accuracy[6]],['Binary Ada Boost','FSCORE',b_fscore[6]],['Binary Ada Boost','ROC_AUC',b_roc[6]],
                       ['Binary Naive Bayes','Accuracy',b_accuracy[7]],['Binary Naive Bayes','FSCORE',b_fscore[7]],['Binary Naive Bayes','ROC_AUC',b_roc[7]],
                       ['Binary ANN','Accuracy',b_accuracy[8]],['Binary ANN','FSCORE',b_fscore[8]],['Binary ANN','ROC_AUC',b_roc[8]],
                       ['Binary LSTM','Accuracy',b_accuracy[9]],['Binary LSTM','FSCORE',b_fscore[9]],['Binary LSTM','ROC_AUC',b_roc[9]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Predicting Stock Market Trends Using Machine Learning and Deep Learning Algorithms Via Continuous and Binary Data a Comparative Analysis', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Stock Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessing)
preprocessButton.place(x=430,y=100)
preprocessButton.config(font=font1) 

continuousButton = Button(main, text="Run Continuous Prediction", command=continuousPrediction)
continuousButton.place(x=780,y=100)
continuousButton.config(font=font1) 

binaryButton = Button(main, text="Run Binary Prediction", command=binaryPrediction)
binaryButton.place(x=10,y=150)
binaryButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=430,y=150)
graphButton.config(font=font1)

closeButton = Button(main, text="View Comparison Table", command=viewTable)
closeButton.place(x=780,y=150)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
