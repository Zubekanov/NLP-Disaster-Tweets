import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('classic')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
    titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

training = pd.read_csv('data/train.csv')

X = training.copy()
y = X.pop('target')

if 'id' in X.columns:
    X = X.drop(columns=['id'])

features_text = ["text"]
features_cat  = ["keyword", "location"]

transformer_text = make_pipeline(
    FunctionTransformer(lambda X: X.squeeze(), validate=False),
    TfidfVectorizer(max_features=10000)
)
transformer_cat  = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='NA'),
    OneHotEncoder(handle_unknown='ignore')
)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75, random_state=42)

preprocessor = make_column_transformer(
    (transformer_text, features_text),
    (transformer_cat,  features_cat),
    remainder='drop'
)

X_train_trans = preprocessor.fit_transform(X_train)
X_valid_trans = preprocessor.transform(X_valid)

input_shape = X_train_trans.shape[1]
model = keras.Sequential([
    layers.InputLayer(shape=(input_shape,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_trans, y_train,
    validation_data=(X_valid_trans, y_valid),
    batch_size=512,
    epochs=200,
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")
plt.show()
