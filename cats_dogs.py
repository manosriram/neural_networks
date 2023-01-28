import cv2
import numpy as np
from tqdm import tqdm
import os
REBUILD_DATA = True

class DogsVsCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print("label = ", self.LABELS[label])
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
                        
                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                            
                    except Exception as e:
                        pass
                    
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print(f"Cats: {self.catcount}")
        print(f"Dogs: {self.dogcount}")

#  if REBUILD_DATA:
    #  dogsvcats = DogsVsCats()
    #  dogsvcats.make_training_data()
                

img = cv2.imread("/Users/manosriram/dev/nn/PetImages/Cat/0.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (50, 50))
img = img / 255.0
print(np.array(img))
