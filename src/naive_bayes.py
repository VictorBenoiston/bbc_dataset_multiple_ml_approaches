import nltk
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.utils import plot_model

# Local imports
from utils import get_dataframe, generate_training_data_matrix, clean_text_in_dataframe

# Avoiding unwanted warnings.
warnings.filterwarnings('ignore')
nltk.download('wordnet')


# Plotting the model
def plot_model_data(model):
    plot_model(model, show_shapes=True, show_layer_names=True)

    history = model.history
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'train loss'], loc='upper right')
    plt.show()


# Classification report
def generate_classification_report(model, label_encoding, inverse_category_label, x_test):
    report = classification_report(
        label_encoding.transform(inverse_category_label),
        model.predict(x_test).argmax(axis=1)
    )
    print(report)


# Confusion Matrix
def generate_confusion_matrix(model, label_encoding, inverse_category_label, x_test):
    cm_rm_2 = confusion_matrix(
        label_encoding.transform(inverse_category_label),
        model.predict(x_test).argmax(axis=1)
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rm_2)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

#########################################

if __name__ == "__main__":
    dataframe = get_dataframe()
    dataframe = clean_text_in_dataframe(dataframe)
    embed_matrix = generate_training_data_matrix(dataframe)

    # Transforming catagories label
    category_label = LabelBinarizer().fit(list(set(dataframe['category'].tolist())))

    X, Y = np.array(embed_matrix), category_label.transform(dataframe['category'].tolist())

    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # label encoding for validation purposes
    label_encoding = LabelEncoder().fit(sorted(list(set(dataframe['category'].tolist()))))
    inverse_category_label = category_label.inverse_transform(np.array(y_test))

    # # GNB Model
    model = GaussianNB()
    model_history = model.fit(x_train, y_train)
    y_preds = model.predict(x_test)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    scores = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')

    plot_model_data(model)
    generate_confusion_matrix(model, label_encoding, inverse_category_label, x_test)
    generate_classification_report(model, label_encoding, inverse_category_label, x_test)
