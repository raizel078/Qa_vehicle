from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import pickle


training_data = [
    # ---------- COUNT ----------
    ("how many cars are in the video", "count"),
    ("count the number of people", "count"),
    ("how many bicycles appeared", "count"),
    ("total number of trucks", "count"),
    ("how many times did a person appear", "count"),
    ("give me the count of cars", "count"),
    ("number of motorcycles in the video", "count"),
    ("how many airplanes are there", "count"),
    ("tell me how many dogs appeared", "count"),
    ("what is the total count of people", "count"),
    ("how many cats in the video", "count"),
    ("count all the buses", "count"),
    ("how many objects were detected", "count"),
    ("total persons detected", "count"),
    ("how many cars did you see", "count"),
    ("sum of all bicycles", "count"),
    ("how many trucks passed by", "count"),
    ("count how many people", "count"),
    ("number of times a car showed up", "count"),
    ("how many phones appeared", "count"),
    ("give the total number of cars", "count"),
    ("how many vehicles altogether", "count"),
    ("count of persons in the footage", "count"),
    ("how many people are present", "count"),
    ("total detections of car", "count"),
    ("how many boats appeared", "count"),
    ("count the cars please", "count"),
    ("how many birds did you detect", "count"),
    ("number of skateboards", "count"),
    ("how many horses were seen", "count"),
    ("count the bikes", "count"),
    ("how many umbrellas in total", "count"),
    ("total number of chairs", "count"),
    ("how many animals altogether", "count"),

    # ---------- LOOKUP ----------
    ("what is at frame 22", "lookup"),
    ("what appeared at 5 seconds", "lookup"),
    ("what objects are at timestamp 10", "lookup"),
    ("what did you detect at frame 40", "lookup"),
    ("what is in frame 15", "lookup"),
    ("show me what happened at 30 seconds", "lookup"),
    ("what was there at timestamp 7", "lookup"),
    ("what appears at the 12 second mark", "lookup"),
    ("what objects at frame 3", "lookup"),
    ("what did the camera see at 45 seconds", "lookup"),
    ("list objects at timestamp 20", "lookup"),
    ("what is present at frame 50", "lookup"),
    ("what happened at second 8", "lookup"),
    ("what was detected at 25 seconds", "lookup"),
    ("what is shown at frame 60", "lookup"),
    ("objects at timestamp 33", "lookup"),
    ("what is at the 2 second point", "lookup"),
    ("what did you find at frame 18", "lookup"),
    ("what was visible at 14 seconds", "lookup"),
    ("what is at timestamp 47", "lookup"),
    ("tell me what is at frame 9", "lookup"),
    ("what showed up at 6 seconds", "lookup"),
    ("what objects appear at 38 seconds", "lookup"),
    ("what is happening at frame 27", "lookup"),
    ("what was at the 4 second mark", "lookup"),
    ("describe frame 11", "lookup"),
    ("what can be seen at timestamp 29", "lookup"),
    ("what is in the video at 16 seconds", "lookup"),
    ("what appeared at frame 44", "lookup"),
    ("what is detected at second 21", "lookup"),
    ("what shows at timestamp 35", "lookup"),
    ("what did you see at frame 5", "lookup"),
    ("what objects were at 50 seconds", "lookup"),

    # ---------- FUZZY ----------
    ("was it busy", "fuzzy"),
    ("was there a lot of traffic", "fuzzy"),
    ("were there any vehicles", "fuzzy"),
    ("did anything interesting happen", "fuzzy"),
    ("was the scene crowded", "fuzzy"),
    ("were there people around", "fuzzy"),
    ("did it look dangerous", "fuzzy"),
    ("was there any activity", "fuzzy"),
    ("were there many objects", "fuzzy"),
    ("did it seem chaotic", "fuzzy"),
    ("was the road busy", "fuzzy"),
    ("were there any animals", "fuzzy"),
    ("was it a quiet scene", "fuzzy"),
    ("were there vehicles present", "fuzzy"),
    ("did people gather", "fuzzy"),
    ("was there congestion", "fuzzy"),
    ("did it look calm", "fuzzy"),
    ("were there any bikes around", "fuzzy"),
    ("was the area empty", "fuzzy"),
    ("did anything unusual appear", "fuzzy"),
    ("was there a crowd", "fuzzy"),
    ("did traffic build up", "fuzzy"),
    ("was it peaceful", "fuzzy"),
    ("were there commuters", "fuzzy"),
    ("did the scene get crowded", "fuzzy"),
    ("was anyone walking", "fuzzy"),
    ("were there transport vehicles", "fuzzy"),
    ("did it feel busy", "fuzzy"),
    ("was there anything happening", "fuzzy"),
    ("were there lots of people", "fuzzy"),
    ("did vehicles pass through", "fuzzy"),
    ("was the street active", "fuzzy"),
    ("did it look like rush hour", "fuzzy"),
]

question = [q for q , label in training_data]
labels = [label for q, label in training_data]

model = SentenceTransformer('all-MiniLM-L6-v2')

x = model.encode(question)
y= labels

clf = LogisticRegression(max_iter=1000)

clf.fit(x,y)

pickle.dump(clf, open('intent_model.pk1','wb'))

