import os

base_path = 'E:\\LLM\\VisualQA'

# Paths for images
train_images_path = os.path.join(base_path, 'Images', 'train2014', 'train2014')
val_images_path = os.path.join(base_path, 'Images', 'val2014', 'val2014')
abstract_train_images_path = os.path.join(base_path, 'AbstractScenes', 'Images', 'Train', 'scene_img_abstract_v002_train2015')

# Paths for annotations
train_annotations_path = os.path.join(base_path, 'Annotations', 'v2_Annotations_Train_mscoco', 'v2_mscoco_train2014_annotations.json')
val_annotations_path = os.path.join(base_path, 'Annotations', 'v2_Annotations_Val_mscoco', 'v2_mscoco_val2014_annotations.json')

# Paths for questions
train_questions_path = os.path.join(base_path, 'Questions', 'v2_Questions_Train_mscoco', 'v2_OpenEnded_mscoco_train2014_questions.json')
val_questions_path = os.path.join(base_path, 'Questions', 'v2_Questions_Val_mscoco', 'v2_OpenEnded_mscoco_val2014_questions.json')

print("Paths consolidated.")
