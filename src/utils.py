import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow_hub as hub

from sklearn.manifold import TSNE

# Local imports
from constants import DATASET_BBC
from preprocessing import clean_text


# Importing packages
EMBED = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# # Data Visualization and Pre-processing
def get_dataframe(path=DATASET_BBC):
    dataframe = pd.read_csv(path)
    dataframe = dataframe[['text', 'category']]

    text_vect = EMBED(dataframe['text'])
    x_reduced = TSNE(n_components=2, random_state=0).fit_transform(text_vect)

    left = pd.concat(
        [pd.DataFrame(x_reduced),dataframe['category']],
        axis=1
    )
    right = (
        pd.DataFrame(dataframe['category'].value_counts())
        .reset_index()
        .rename(columns={"index":"category"})
    )
    pd.merge(
        left = left,
        right = right,
        on='category', how='left'
    )

    return dataframe


#cleaning text in data frames
def clean_text_in_dataframe(dataframe):
    dataframe["textclean"] = dataframe["text"].apply(clean_text)
    dataframe[["textclean", "text", "category"]].iloc[90]

    textclean, text = dataframe[["textclean", "text", "category"]].iloc[90]['textclean'], dataframe[["textclean", "text", "category"]].iloc[90]['text']
    textclean, text = EMBED([textclean]), EMBED([text])
    matrix = np.corrcoef(textclean, text)
    sns.heatmap(matrix, annot=True)
    return dataframe


# generating training data matrix
def generate_training_data_matrix(dataframe):
    sentences = dataframe["textclean"].apply(lambda x: x.lower()).tolist()
    embed_matrix = []
    for sent in sentences:
        embed_matrix.append(np.array(EMBED([sent])[0]).tolist())
    return embed_matrix
