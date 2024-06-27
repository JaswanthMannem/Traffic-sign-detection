import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM
from keras.layers import Reshape, TimeDistributed
from keras import backend as K
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# Load labels
labels_df = pd.read_csv('pythonProject/archive(1)/labels.csv')

# Load images and labels
def load_images_and_labels(image_folder_path, labels_df):
    images = []
    labels = []
    for folder_name in os.listdir(image_folder_path):
        class_id = int(folder_name)
        image_path = os.path.join(image_folder_path, folder_name)
        for image_file in glob.glob(image_path + '/*.jpg'):
            image = load_image(image_file)
            images.append(image)
            labels.append(labels_df[labels_df['ClassId'] == class_id]['Name'].values[0])
    im=np.array(images)
    lb=np.array(labels)
    print(im)
    print(lb)
    return np.array(images), np.array(labels)

# Function to load and preprocess an image
def load_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess an image.

    Parameters:
    - image_path: The path to the image file.
    - target_size: The desired size of the image as a tuple (width, height).

    Returns:
    A preprocessed image suitable for model input.
    """
    # Open the image file
    img = Image.open(image_path)
    # Resize the image to the target size
    img = img.resize(target_size)
    # Convert the image to an array and normalize to the range [0, 1]
    img = img_to_array(img) / 255.0
    return img

# Define the autoencoder models
from keras.layers import Flatten, Dense


def autoencoder_cnn(input_shape, num_classes):
    input_img = Input(shape=input_shape)  # Placeholder for input
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # Create the model
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Additional layers for classification
    decoded = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)  # Last conv layer
    x = Flatten()(decoded)
    x = Dense(num_classes, activation='softmax')(x)  # Output layer for multi-class



    # Flatten the encoded output for Dense layers
    x = Flatten()(encoded)
    x = Dense(128, activation='relu')(x)  # Example additional dense layer

    # Output layer for classification
    classification_output = Dense(num_classes, activation='softmax')(x)

    # Create the model
    autoencoder = Model(input_img, classification_output)

    return autoencoder


from keras.layers import Input, SimpleRNN, RepeatVector, TimeDistributed, Reshape
from keras.models import Model


def autoencoder_rnn(input_shape, num_classes):
    input_img = Input(shape=input_shape)

    height, width, channels = input_shape
    sequence_length = height
    sequence_width = width * channels

    # Encoder
    encoded = Reshape((sequence_length, sequence_width))(input_img)
    encoded = SimpleRNN(128, activation='relu', return_sequences=False)(encoded)

    # Classification layers
    x = Dense(64, activation='relu')(encoded)
    classification_output = Dense(num_classes, activation='softmax')(x)

    # Create model
    autoencoder = Model(input_img, classification_output)
    return autoencoder



from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model


def autoencoder_lstm(input_shape, num_classes):
    # Assuming input_shape is in the form [height, width, channels],
    # we convert the image into a sequence where each row is considered
    # a timestep, and the features are the pixel values in that row.
    timesteps = input_shape[0]  # height as timesteps
    input_dim = input_shape[1] * input_shape[2]  # width * channels as features

    # Encoder
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(128)(inputs)

    # Classification layers
    x = Dense(64, activation='relu')(encoded)
    classification_output = Dense(num_classes, activation='softmax')(x)

    # Autoencoder Model
    autoencoder = Model(inputs, classification_output)
    return autoencoder



# Compile and train the models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


def compile_and_train(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    """
    Compile and train a given Keras model.

    Parameters:
    - model: The Keras model to train.
    - X_train: Training data.
    - y_train: Labels for training data.
    - X_val: Validation data.
    - y_val: Labels for validation data.
    - batch_size: Number of samples per gradient update.
    - epochs: Number of epochs to train the model.

    Returns:
    The history object containing recorded training and validation statistics.
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping and model checkpoint for efficiency and to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('model_best.keras', monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    return history


# Evaluate models and generate metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import seaborn as sns



def evaluate_model(model, X_test, y_test, class_labels):
    """
    Evaluate the model on the test set and generate classification metrics,
    including the confusion matrix and ROC curve.

    Parameters:
    - model: The trained Keras model to evaluate.
    - X_test: Test data.
    - y_test: True labels for the test data.
    - class_labels: List of class labels that correspond to the output of the model.

    Returns:
    A dictionary containing the evaluation metrics.
    """
    # Predict the probabilities for each class
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_test_class = np.argmax(y_test, axis=1)

    # Calculate metrics
    confusion = confusion_matrix(y_test_class, y_pred_class)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class, average='macro')
    recall = recall_score(y_test_class, y_pred_class, average='macro')
    f1 = f1_score(y_test_class, y_pred_class, average='macro')

    # Compute ROC curve and ROC area for each class
    y_test_binarized = label_binarize(y_test_class, classes=range(len(class_labels)))
    n_classes = y_test_binarized.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {model.name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # Return the metrics
    metrics = {
        'Confusion Matrix': confusion,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

    return metrics


# Prediction function
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


def predict_image(model, image_path, target_size=(32, 32)):
    """
    Predict the category of the given image using the trained model.

    Parameters:
    - model: The trained Keras model to use for prediction.
    - image_path: The file path to the image to predict.
    - target_size: The expected input size of the model (width, height).

    Returns:
    The predicted category as an integer index.
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Model expects a batch of images as input
    img = img / 255.0  # Assuming model was trained on images scaled to [0, 1]

    # Predict the class
    pred = model.predict(img)
    predicted_class = np.argmax(pred, axis=1)

    return predicted_class[0]  # Return the predicted class index


# Example usage:
# Assuming `model` has already been trained and `image_path` is the path to the new image
# predicted_class = predict_image(model, image_path)
# print(f"The image is predicted to be class index: {predicted_class}")


# Main execution
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
def main():
    # Path to your image folders and labels
    image_folder_path = 'pythonProject/archive(1)/myData'
    labels_csv_path = 'pythonProject/archive(1)/labels.csv'

    # Load labels
    labels_df = pd.read_csv(labels_csv_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_df['Name'])
    # Load images and labels
    images, labels = load_images_and_labels(image_folder_path, labels_df)
    labels = label_encoder.transform(labels)
    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Reshape data for LSTM
    X_train_reshaped = X_train.reshape((-1, X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    X_val_reshaped = X_val.reshape((-1, X_val.shape[1], X_val.shape[2] * X_val.shape[3]))
    X_test_reshaped = X_test.reshape((-1, X_test.shape[1], X_test.shape[2] * X_test.shape[3]))

    # Preprocess labels
    num_classes = len(np.unique(labels))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Define model input shape
    input_shape = X_train[0].shape

    # Define and train CNN Autoencoder
    cnn_autoencoder = autoencoder_cnn(input_shape, num_classes)
    print("Training and evaluating CNN Autoencoder")
    compile_and_train(cnn_autoencoder, X_train, y_train, X_val, y_val)
    cnn_metrics = evaluate_model(cnn_autoencoder, X_test, y_test, class_labels=labels_df['Name'].unique())
    print("CNN Autoencoder metrics:", cnn_metrics)

    # Define and train RNN Autoencoder
    rnn_autoencoder = autoencoder_rnn(input_shape, num_classes)
    print("Training and evaluating RNN Autoencoder")
    compile_and_train(rnn_autoencoder, X_train, y_train, X_val, y_val)
    rnn_metrics = evaluate_model(rnn_autoencoder, X_test, y_test, class_labels=labels_df['Name'].unique())
    print("RNN Autoencoder metrics:", rnn_metrics)

    # Define and train LSTM Autoencoder
    lstm_autoencoder = autoencoder_lstm(input_shape, num_classes)
    print("Training and evaluating LSTM Autoencoder")
    compile_and_train(lstm_autoencoder, X_train_reshaped, y_train, X_val_reshaped, y_val)
    lstm_metrics = evaluate_model(lstm_autoencoder, X_test_reshaped, y_test, class_labels=labels_df['Name'].unique())
    print("LSTM Autoencoder metrics:", lstm_metrics)

    # Prediction example using CNN Autoencoder
    image_path = 'pythonProject/archive(1)/myData/2/00000_00000.jpg'
    prediction_index = predict_image(cnn_autoencoder, image_path)
    prediction_label = labels_df['Name'][prediction_index]
    print(f'Predicted category for the image is: {prediction_label}')

    # For each model, update results and print metrics

if __name__ == "__main__":
    main()