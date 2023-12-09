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

def load_images(directory, limit=100):
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if i >= limit:
            break
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = load_and_preprocess_image(img_path)
            images.append(img)
    return np.array(images)

def load_questions_and_annotations(question_file, annotation_file):
    with open(question_file, 'r') as qf, open(annotation_file, 'r') as af:
        questions = json.load(qf)['questions']
        annotations = json.load(af)['annotations']

    question_texts = [q['question'] for q in questions]
    question_ids = [q['question_id'] for q in questions]
    answers = [a['multiple_choice_answer'] for a in annotations]

    # Tokenize questions
    encoded_questions = tokenizer(question_texts, padding=True, truncation=True, max_length=32, return_tensors='np').input_ids

    return question_ids, encoded_questions, answers

# Load and preprocess the data
train_images = load_images('E:/LLM/VisualQA/Images/train2014/train2014')
val_images = load_images('E:/LLM/VisualQA/Images/val2014/val2014')

train_question_ids, train_encoded_questions, train_answers = load_questions_and_annotations(
    'E:/LLM/VisualQA/Questions/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json', 
    'E:/LLM/VisualQA/Annotations/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json'
)

val_question_ids, val_encoded_questions, val_answers = load_questions_and_annotations(
    'E:/LLM/VisualQA/Questions/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json', 
    'E:/LLM/VisualQA/Annotations/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json'
)

# Prepare the model (placeholder for actual implementation)
num_classes = 1000  # Placeholder, adjust this based on your actual number of possible answers
base_image_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_input = Input(shape=(224, 224, 3), dtype=tf.float32)
x_image = base_image_model(preprocess_input(image_input))
x_image = GlobalAveragePooling2D()(x_image)

distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

max_seq_length = 32
question_input_ids = Input(shape=(max_seq_length,), dtype=tf.int32)
question_mask = Input(shape=(max_seq_length,), dtype=tf.int32)

distilbert_output = distilbert_model(question_input_ids, attention_mask=question_mask)
x_question = distilbert_output.last_hidden_state
x_question = GlobalAveragePooling1D()(x_question)

combined = Concatenate()([x_image, x_question])
dense = Dense(512, activation='relu')(combined)
output = Dense(num_classes, activation='softmax')(dense)

model = Model(inputs=[image_input, question_input_ids, question_mask], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming that 'train_images', 'val_images', 'train_encoded_questions', and 'val_encoded_questions' are already loaded from Code 1

# Model Training
model.fit(
    [train_images, train_encoded_questions.input_ids, train_encoded_questions.attention_mask], train_answers,
    validation_data=([val_images, val_encoded_questions.input_ids, val_encoded_questions.attention_mask], val_answers),
    epochs=10,  # You can adjust the number of epochs
    batch_size=32  # And the batch size
)
