import os
import pickle
from deepface import DeepFace

dataset_path = "datasets"

embeddings = []
names = []

# loop on persons
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    # loop on images
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            # extract embedding using FaceNet
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                enforce_detection=False
            )

            embeddings.append(embedding[0]["embedding"])
            names.append(person_name)

            print(f"Added: {person_name} - {image_name}")

        except Exception as e:
            print(f"Error in {image_path}")
            print(e)

# save trained data
data = {
    "embeddings": embeddings,
    "names": names
}

with open("facenet_trained.pkl", "wb") as f:
    pickle.dump(data, f)

print("Training Finished")
