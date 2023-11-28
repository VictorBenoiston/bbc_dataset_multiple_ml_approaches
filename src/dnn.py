import nltk
import visualkeras
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from itertools import product
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.utils import plot_model

# Local imports
from utils import get_dataframe, generate_training_data_matrix, clean_text_in_dataframe

# Avoiding unwanted warnings.
warnings.filterwarnings('ignore')
nltk.download('wordnet')


def train_model(x_train, y_train):
    learning_rates = [0.001, 0.0001, 0.00001]
    units_values = [64, 128, 256]
    epochs_values = [5, 20, 40]

    # Creating all possible combinations of hyperparameters
    hyperparameter_combinations = list(product(learning_rates, units_values, epochs_values))

    input_dim = 512

    # Dictionary to store training results for each combination
    all_results = {}

    # Training the models for each possible combination
    for learning_rate, units, epochs in hyperparameter_combinations:
        # Model Architecture
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.Input(shape=input_dim))
        ann.add(tf.keras.layers.Dense(units=units, activation='relu'))
        ann.add(tf.keras.layers.Dropout(0.5))
        ann.add(tf.keras.layers.Dense(units=5, activation='softmax'))

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        ann.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = ann.fit(x_train, y_train, epochs=epochs, batch_size=10, validation_split=0.20)

        # Get the final training loss and accuracy
        final_train_loss = history.history['loss'][-1]
        final_train_accuracy = history.history['accuracy'][-1]

        # Get the validation loss and accuracy if available
        if 'val_loss' in history.history and 'val_accuracy' in history.history:
            final_val_loss = history.history['val_loss'][-1]
            final_val_accuracy = history.history['val_accuracy'][-1]
        else:
            final_val_loss = None
            final_val_accuracy = None

        # Store relevant information in the dictionary
        model_key = f"LR_{learning_rate}_Units_{units}_Epochs_{epochs}"
        all_results[model_key] = {
            'learning_rate': learning_rate,
            'units': units,
            'epochs': epochs,
            'final_train_loss': final_train_loss,
            'final_train_accuracy': final_train_accuracy,
            'final_val_loss': final_val_loss,
            'final_val_accuracy': final_val_accuracy
        }

    return ann, all_results


# Showing all the models parameters for analysis.
def show_models_parameters(results):
    for k, va in results.items():
        print(f'Architecture {k} parameters: ')
        for k, v in va.items():
            print(f'{k}: {v:.6f}')
    print('-----------')


def build_model(seed=None):
  tf.random.set_seed(seed)

  best_ann = tf.keras.models.Sequential()
  best_ann.add(tf.keras.Input(shape=512))
  best_ann.add(tf.keras.layers.Dense(units=256, activation='relu'))
  best_ann.add(tf.keras.layers.Dropout(0.5))
  best_ann.add(tf.keras.layers.Dense(units=5, activation='softmax'))
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  best_ann.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics ='categorical_accuracy')
  model = best_ann
  return model


# Testing 5 different seeds (5 different initial values, hence, 5 fits)
def get_tested_models(x_train, y_train, X, Y):
    seeds = [1, 2, 3, 4, 5]  # For the sake of reproducibility
    models = {}

    for seed in seeds:
        model = build_model(seed)
        Kmodel = KerasClassifier(build_fn=build_model, batch_size=10, verbose=1)

        model_history = model.fit(x_train, y_train, epochs=40, batch_size=10, validation_split=0.20)

        # Implementing the 10-fold cross validation
        scores = cross_val_score(Kmodel, X, Y, cv=10, scoring='accuracy')


        # Get the final training loss and accuracy
        final_train_loss = model_history.history['loss'][-1]
        final_train_accuracy = model_history.history['categorical_accuracy'][-1]

        # Get the validation loss and accuracy if available
        # if 'val_loss' in history.history and 'val_accuracy' in history.history:
        final_val_loss = model_history.history['val_loss'][-1]
        final_val_accuracy = model_history.history['val_categorical_accuracy'][-1]
        # else:
        #     final_val_loss = None
        #     final_val_accuracy = None

        model_key = f"best_architecture_seed_{seed}"
        models[model_key] = {
            'final_train_loss': f'{final_train_loss:.4f}',
            'final_train_accuracy': f'{final_train_accuracy:.4f}',
            'final_val_loss': f'{final_val_loss:.4f}',
            'final_val_accuracy': f'{final_val_accuracy:.4f}',
            '10_fold_average': f'{scores.mean():.4f}',
            '10_fold_standard deviation': f'{scores.std():.4f}',
            'seed': seed
            }

        # Cleaning the variable
        model_history = ''
        Kmodel = ''

    return models


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

    # DNN Model

    ann, all_results = train_model(x_train, y_train)
    visualkeras.layered_view(ann,legend=True, draw_volume=True, spacing=30)
    show_models_parameters(all_results)
    # Best model architecture:

    # Architecture LR_0.0001_Units_256_Epochs_40 parameters:
    # learning_rate: 0.000100
    # units: 256.000000
    # epochs: 40.000000
    # final_train_loss: 0.062563
    # final_train_accuracy: 0.989667
    # final_val_loss: 0.075204
    # final_val_accuracy: 0.986239


    # Checking the models
    models = get_tested_models(x_train, y_train, X, Y)

    # After several tests, we ended up with the best model Architecture LR_0.0001_Units_256_Epochs_40 with seed 3.
    model = build_model(3)
    model_history = model.fit(x_train, y_train, epochs=40, batch_size=10, validation_split=0.20)
    model.summary()

    plot_model_data(model)
    generate_confusion_matrix(model, label_encoding, inverse_category_label, x_test)
    generate_classification_report(model, label_encoding, inverse_category_label, x_test)