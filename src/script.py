
# Importing packages
import os
import nltk
import platform
import re
import string
import visualkeras
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub

from itertools import product
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scikeras.wrappers import KerasClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.utils import plot_model
from wordcloud import WordCloud, STOPWORDS


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# # Data Visualization and Pre-processing

df = pd.read_csv("bbc-text.csv")
df = df[['text', 'category']]

df.describe(include='all')

# Avoiding unwanted warnings.
warnings.filterwarnings('ignore')

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

text_vect = embed(df['text'])
x_reduced = TSNE(n_components=2, random_state=0).fit_transform(text_vect)

left = pd.concat([pd.DataFrame(x_reduced),df['category']],axis = 1)
print(left.head())
right = pd.DataFrame(df['category'].value_counts()).reset_index().rename(columns={"index":"category"})
print(right.head())
pd.merge(
    left = left,
    right = right,
    on='category', how='left')

# Transforming the most used abbraviations.

abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk",
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet",
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously",
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

#Generating Word Coud

token = list(abbreviations.keys())
comment_words = ''
stopWords = set(STOPWORDS)


for i in range(len(token)):
    token[i] = token[i].lower()

    comment_words += " ".join(token)+" "

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopWords,
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()

# Remove all URLs, replace by URL
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'URL',text)


# Remove HTML beacon
def remove_HTML(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Remove non printable characters
def remove_not_ASCII(text):
    text = ''.join([word for word in text if word in string.printable])
    return text

# Change an abbreviation by its true meaning
def word_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

# Replace all abbreviations
def replace_abbrev(text):
    string = ""
    for word in text.split():
        string += word_abbrev(word) + " "
    return string

# Remove @ and mention, replace by USER
def remove_mention(text):
    at=re.compile(r'@\S+')
    return at.sub(r'USER',text)


# Remove numbers, replace it by NUMBER
def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)



# Replace some others smileys with SADFACE
def transcription_sad(text):
    eyes = "[8:=;]"
    nose = "['`\-]"
    smiley = re.compile(r'[8:=;][\'\-]?[(\\/]')
    return smiley.sub(r'SADFACE', text)


# Replace some smileys with SMILE
def transcription_smile(text):
    eyes = "[8:=;]"
    nose = "['`\-]"
    smiley = re.compile(r'[8:=;][\'\-]?[)dDp]')
    #smiley = re.compile(r'#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i')
    return smiley.sub(r'SMILE', text)


# Replace <3 with HEART
def transcription_heart(text):
    heart = re.compile(r'<3')
    return heart.sub(r'HEART', text)


# Factorize elongated words, add ELONG
def remove_elongated_words(text):
    rep = re.compile(r'\b(\S*?)([a-z])\2{2,}\b')
    return rep.sub(r'\1\2 ELONG', text)


# Factorize repeated punctuation, add REPEAT
def remove_repeat_punct(text):
    rep = re.compile(r'([!?.]){2,}')
    return rep.sub(r'\1 REPEAT', text)



# Remove all punctuations
def remove_all_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)


# Remove punctuations
def remove_punct(text):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        text = text.replace(p, f' {p} ')

    text = text.replace('...', ' ... ')
    if '...' not in text:
        text = text.replace('..', ' ... ')
    return text


# Remove all english stopwords
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text

def stemming(text):
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatization(text):
    lm= WordNetLemmatizer()
    text = ' '.join([lm.lemmatize(word, pos='v') for word in text.split()])
    return text

def clean_text(text):

    # Remove non text
    text = remove_URL(text)
    text = remove_HTML(text)
    text = remove_not_ASCII(text)

    # Lower text, replace abbreviations
    text = replace_abbrev(text)
    text = remove_mention(text)
    text = remove_number(text)

    # Remove  smileys

    text = transcription_sad(text)
    text = transcription_smile(text)
    text = transcription_heart(text)

    # Remove repeated puntuations / words
    text = remove_elongated_words(text)
    text = remove_repeat_punct(text)

    text = remove_all_punct(text)
    text = remove_punct(text)
    text = lemmatization(text)
    text = text.lower()

    return text


nltk.download('wordnet')

#cleaning text in data frames
df["textclean"] = df["text"].apply(clean_text)

df[["textclean", "text", "category"]].iloc[90]

textclean, text = df[["textclean", "text", "category"]].iloc[90]['textclean'], df[["textclean", "text", "category"]].iloc[90]['text']
textclean, text = embed([textclean]), embed([text])
matrix = np.corrcoef(textclean, text)
sns.heatmap(matrix, annot=True)

sentences = df["textclean"].apply(lambda x: x.lower()).tolist()
# generating training data matrix
embed_matrix = []
for sent in sentences:
    embed_matrix.append(np.array(embed([sent])[0]).tolist())

# Transforming catagories label
lb = LabelBinarizer().fit(list(set(df['category'].tolist())))

X, Y = np.array(embed_matrix), lb.transform(df['category'].tolist())

x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


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
    if platform.processor() == 'arm':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
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

# Showing all the models parameters for analysis.

for k, va in all_results.items():
  print(f'Architecture {k} parameters: ')
  for k, v in va.items():
    print(f'{k}: {v:.6f}')
  print('-----------')

# Best model architecture:

# Architecture LR_0.0001_Units_256_Epochs_40 parameters:
# learning_rate: 0.000100
# units: 256.000000
# epochs: 40.000000
# final_train_loss: 0.062563
# final_train_accuracy: 0.989667
# final_val_loss: 0.075204
# final_val_accuracy: 0.986239

def build_model(seed=None):
  tf.random.set_seed(seed)

  best_ann = tf.keras.models.Sequential()
  best_ann.add(tf.keras.Input(shape=512))
  best_ann.add(tf.keras.layers.Dense(units=256, activation='relu'))
  best_ann.add(tf.keras.layers.Dropout(0.5))
  best_ann.add(tf.keras.layers.Dense(units=5, activation='softmax'))
  
  learning_rate=0.0001
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  if platform.processor() == 'arm':
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

  best_ann.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics ='categorical_accuracy')
  model = best_ann
  return model


# Testing 5 different seeds (5 different initial values, hence, 5 fits)

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

# Best model with seed = 1:
# model = build_model(1)

# best_model = build_model()

# Kmodel = KerasClassifier(build_fn=build_model, verbose=1)

# Checking the models
models

# After several tests, we ended up with the best model Architecture LR_0.0001_Units_256_Epochs_40 with seed 3.

model = build_model(3)

model_history = model.fit(x_train, y_train, epochs=40, batch_size=10, validation_split=0.20)

# Summary
model.summary()


visualkeras.layered_view(ann,legend=True, draw_volume=True, spacing=30)

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

# label encoding for validation purposes
le = LabelEncoder().fit(sorted(list(set(df['category'].tolist()))))

# Confusion Matrix
cm_rm_2 = confusion_matrix(le.transform(lb.inverse_transform(np.array(y_test))),model.predict(x_test).argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rm_2)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print(classification_report(le.transform(lb.inverse_transform(np.array(y_test))),model.predict(x_test).argmax(axis=1)))


