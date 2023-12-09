import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.preprocessing import LabelEncoder

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def load_and_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
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

def load_questions_and_annotations(question_file, annotation_file, limit=100):
    with open(question_file, 'r') as qf, open(annotation_file, 'r') as af:
        questions = json.load(qf)['questions']
        annotations = json.load(af)['annotations']

    questions = questions[:limit]
    annotations = annotations[:limit]

    question_texts = [q['question'] for q in questions]
    encoded_questions = tokenizer(question_texts, padding=True, truncation=True, max_length=32, return_tensors='tf')
    answers = [a['multiple_choice_answer'] for a in annotations]

    return encoded_questions, answers

# Load and preprocess the data
train_images = load_images('E:/LLM/VisualQA/Images/train2014/train2014', limit=100)
val_images = load_images('E:/LLM/VisualQA/Images/val2014/val2014', limit=100)
train_encoded_questions, train_answers = load_questions_and_annotations(
    'E:/LLM/VisualQA/Questions/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json',
    'E:/LLM/VisualQA/Annotations/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json',
    limit=100
)
val_encoded_questions, val_answers = load_questions_and_annotations(
    'E:/LLM/VisualQA/Questions/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json',
    'E:/LLM/VisualQA/Annotations/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json',
    limit=100
)

# Convert answers to numerical labels and one-hot encode them
label_encoder = LabelEncoder()
label_encoder.fit(np.concatenate([train_answers, val_answers]))
encoded_train_answers = label_encoder.transform(train_answers)
encoded_val_answers = label_encoder.transform(val_answers)

num_classes = len(label_encoder.classes_)
train_answers_one_hot = tf.keras.utils.to_categorical(encoded_train_answers, num_classes=num_classes)
val_answers_one_hot = tf.keras.utils.to_categorical(encoded_val_answers, num_classes=num_classes)

# Model definition
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

# Ensure all input data are in the correct format
train_images = np.array(train_images)
val_images = np.array(val_images)
train_input_ids = tf.convert_to_tensor(train_encoded_questions['input_ids'])
train_attention_mask = tf.convert_to_tensor(train_encoded_questions['attention_mask'])
val_input_ids = tf.convert_to_tensor(val_encoded_questions['input_ids'])
val_attention_mask = tf.convert_to_tensor(val_encoded_questions['attention_mask'])

# Model Training
model.fit(
    [train_images, train_input_ids, train_attention_mask], train_answers,
    validation_data=([val_images, val_input_ids, val_attention_mask], val_answers),
    epochs=10,  # Adjust the number of epochs as needed
    batch_size=32  # Adjust the batch size as needed
)

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(
    [val_images, val_input_ids, val_attention_mask], val_answers
)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the model
model.save("E:/LLM/your_model1.h5")