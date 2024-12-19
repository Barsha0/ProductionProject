#importing basic packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# Random Forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model
from sklearn.svm import SVC
import pickle
import os
import joblib
import pandas as pd
from urllib.parse import urlparse
import re



filename = 'XGBoostClassifier.pickle.dat'

if(not os.path.exists(filename)):
    data = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\Production\\Dataset\\urldata.csv")
    data.head()

    #Dropping the Domain column
    dfsa = data.drop(['Domain'], axis = 1).copy()
    dfsa.isnull().sum()
    dfsa = dfsa.sample(frac=1).reset_index(drop=True)
    dfsa.head()

    # Sepratating & assigning features and target columns to X & y
    y = dfsa['Label']  #target variable
    X = dfsa.drop('Label',axis=1)   #independent variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)  #test-size 20%
    X_train.shape, X_test.shape

    # Creating holders to store the model performance results
    ML_Model = []
    acc_train = []
    acc_test = []

    #function to call for storing the results
    def storeResults(model, a,b):
        ML_Model.append(model)
        acc_train.append(round(a, 3))
        acc_test.append(round(b, 3))


    tree = DecisionTreeClassifier(max_depth = 5)
    # fit the model 
    tree.fit(X_train, y_train)

    #predicting the target value from the model for the samples
    y_test_tree = tree.predict(X_test)
    y_train_tree = tree.predict(X_train)

    #computing the accuracy of the model performance
    acc_train_tree = accuracy_score(y_train,y_train_tree)
    acc_test_tree = accuracy_score(y_test,y_test_tree)

    print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
    print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

    

    # instantiate the model
    forest = RandomForestClassifier(max_depth=5)

    # fit the model 
    forest.fit(X_train, y_train)

    #predicting the target value from the model for the samples
    y_test_forest = forest.predict(X_test)
    y_train_forest = forest.predict(X_train)

    #computing the accuracy of the model performance
    acc_train_forest = accuracy_score(y_train,y_train_forest)
    acc_test_forest = accuracy_score(y_test,y_test_forest)

    print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
    print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

    import matplotlib.pyplot as plt

    # Plot accuracy for training and test data
    plt.figure(figsize=(8, 6))
    plt.bar(['Training', 'Test'], [acc_train_forest, acc_test_forest], color=['blue', 'green'])
    plt.title('Accuracy of Random Forest Model')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
    plt.show()
    

    # instantiate the model
    mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))

    # fit the model 
    mlp.fit(X_train, y_train)

    #predicting the target value from the model for the samples
    y_test_mlp = mlp.predict(X_test)
    y_train_mlp = mlp.predict(X_train)

    # instantiate the model
    xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
    #fit the model
    xgb.fit(X_train, y_train)
    #predicting the target value from the model for the samples
    y_test_xgb = xgb.predict(X_test)
    y_train_xgb = xgb.predict(X_train)

    #building autoencoder model

    input_dim = X_train.shape[1]
    encoding_dim = input_dim

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="relu",
                    activity_regularizer=regularizers.l1(10e-4))(input_layer)
    encoder = Dense(int(encoding_dim), activation="relu")(encoder)

    encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
    code = Dense(int(encoding_dim-4), activation='relu')(encoder)
    decoder = Dense(int(encoding_dim-2), activation='relu')(code)

    decoder = Dense(int(encoding_dim), activation='relu')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    #compiling the model
    autoencoder.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

    #Training the model
    history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2) 

    acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
    acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

    print('\nAutoencoder: Accuracy on training Data: {:.3f}' .format(acc_train_auto))
    print('Autoencoder: Accuracy on test Data: {:.3f}' .format(acc_test_auto))

    # instantiate the model
    svm = SVC(kernel='linear', C=1.0, random_state=12)
    #fit the model
    svm.fit(X_train, y_train)

    #predicting the target value from the model for the samples
    y_test_svm = svm.predict(X_test)
    y_train_svm = svm.predict(X_train)
    pickle.dump(xgb, open("XGBoostClassifier.pickle.dat", "wb"))


    # load model from file
    loaded_model = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))
    loaded_model

# Define feature extraction
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'Have_IP': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        'Have_At': int('@' in parsed_url.netloc),
        'URL_Length': len(url),
        'URL_Depth': urlparse(url).path.count('/'),
        'Redirection': int('//' in urlparse(url).path),
        'https_Domain': int(parsed_url.scheme == 'https'),
        'TinyURL': int(len(url) < 20),
        'Prefix/Suffix': int('-' in parsed_url.netloc),
        'DNS_Record': 1,  # Placeholder, replace with actual DNS record check
        'Web_Traffic': 1,  # Placeholder, replace with actual web traffic data
        'Domain_Age': 1,   # Placeholder, replace with actual domain age
        'Domain_End': 1,   # Placeholder, replace with actual domain end date
        'iFrame': 1,       # Placeholder, replace with actual iFrame check
        'Mouse_Over': 1,   # Placeholder, replace with actual mouse over check
        'Right_Click': 1,  # Placeholder, replace with actual right-click check
        'Web_Forwards': 1  # Placeholder, replace with actual web forwards check
    }
    return features

# Load model
model = joblib.load('XGBoostClassifier.pickle.dat')


# Prepare input data
def prepare_input_data(features):
    features_df = pd.DataFrame([features])
    return features_df

# Prediction function
def predict_phishing(url):
    features = extract_features(url)
    features_df = prepare_input_data(features)
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)
    return prediction[0], probability[0]

def isSafeUrl(url):
    prediction, probability = predict_phishing(url)
    print(f"Prediction: {'Phishing' if prediction == 1 else 'Not Phishing'}")
    print(f"Probability: {probability}")
    return prediction != 1

#print(isSafeUrl("https://www.google.com"))







