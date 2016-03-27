import json
import os
from random import shuffle

class Face:
   @classmethod
   def from_file_name(self, file_name):
      face_name = file_name[0:2]
      emotion = file_name[3:5]
      file_name = file_name
      return Face(face_name=face_name, emotion=emotion, file_name=file_name)

   def __init__(self, face_name, emotion, file_name):
      self.face_name = face_name
      self.emotion = emotion
      self.file_name = file_name

class Index:
   def __init__(self, training_set, testing_set):
      self.training_set = training_set
      self.testing_set = testing_set

def create_index(jaffe_dir):
   emotion_to_faces_dict = {
      'AN': [],
      'DI': [],
      'FE': [],
      'HA': [],
      'NE': [],
      'SA': [],
      'SU': []
   }

   for root, dirs, files in os.walk(jaffe_dir, topdown=False):
      for name in files:
         if os.path.splitext(name)[1].lower() == '.png':
            face = Face.from_file_name(name)
            emotion_to_faces_dict[face.emotion].append(face)

   for faces in emotion_to_faces_dict.itervalues():
      shuffle(faces)

   training_set = []
   testing_set = []
   for faces in emotion_to_faces_dict.itervalues():
      testing_set += faces[:3]
      training_set += faces[3:]

   index = Index(training_set, testing_set)
   with open(os.path.join(jaffe_dir, 'index.json'), 'w+') as outfile:
      json.dump(index.__dict__, outfile, default=lambda obj: obj.__dict__, indent=3)

def load_index(jaffe_dir):
   with open(os.path.join(jaffe_dir, 'index.json'), 'r') as infile:
      index_json = json.load(infile)

   def face_decoder(obj):
      return Face(face_name=obj['face_name'], emotion=obj['emotion'], file_name=obj['file_name'])

   training_set = map(face_decoder, index_json['training_set'])
   testing_set = map(face_decoder, index_json['testing_set'])
   return Index(training_set=training_set, testing_set=testing_set)

if __name__ == '__main__':
   create_index('jaffetest')
