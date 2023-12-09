import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import DistilBertTokenizer, TFDistilBertModel

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def load_and_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = preprocess_input(img)  # Preprocess for MobileNetV2
    return img

# pre-trained MobileNetV2 model for image feature extraction
base_image_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_input = Input(shape=(224, 224, 3), dtype=tf.float32)
x_image = base_image_model(preprocess_input(image_input))
x_image = GlobalAveragePooling2D()(x_image)

# pre-trained DistilBERT model for text feature extraction
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Assumtion maximum sequence length for questions is 32
max_seq_length = 32
question_input_ids = Input(shape=(max_seq_length,), dtype=tf.int32)
question_mask = Input(shape=(max_seq_length,), dtype=tf.int32)

distilbert_output = distilbert_model(question_input_ids, attention_mask=question_mask)
x_question = distilbert_output.last_hidden_state
x_question = GlobalAveragePooling1D()(x_question)

# Combining features from both models
combined = Concatenate()([x_image, x_question])

# Add dense layers for combined features
dense = Dense(512, activation='relu')(combined)
output = Dense(1000, activation='softmax')(dense)  # Using a placeholder value for num_classes

# Build model
model = Model(inputs=[image_input, question_input_ids, question_mask], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Model Training
model.fit(
    [train_images, train_encoded_questions.input_ids, train_encoded_questions.attention_mask], train_answers,
    validation_data=([val_images, val_encoded_questions.input_ids, val_encoded_questions.attention_mask], val_answers),
    epochs=10,  # You can adjust the number of epochs
    batch_size=32  # And the batch size
)