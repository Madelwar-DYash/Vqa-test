import os
import json
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from transformers import DistilBertTokenizer

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def load_and_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = preprocess_input(img)  # Preprocess for MobileNetV2
    return img

def load_images(directory, limit=1000):
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if i >= limit:  # Load only a limited number of images
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
    encoded_questions = tokenizer(question_texts, padding=True, truncation=True, return_tensors='np')

    return question_ids, encoded_questions, answers

# Example usage:
train_images = load_images('E:/LLM/VisualQA/Images/train2014/train2014', limit=100)  # Adjust the limit as needed
val_images = load_images('E:/LLM/VisualQA/Images/val2014/val2014', limit=100)  # Adjust the limit as needed

train_question_ids, train_encoded_questions, train_answers = load_questions_and_annotations(
    'E:/LLM/VisualQA/Questions/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json', 
    'E:/LLM/VisualQA/Annotations/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json'
)

val_question_ids, val_encoded_questions, val_answers = load_questions_and_annotations(
    'E:/LLM/VisualQA/Questions/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json', 
    'E:/LLM/VisualQA/Annotations/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json'
)
