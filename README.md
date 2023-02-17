# classup
## Joshua

## Ryo
> AI Features
## Facial Recognition Attendance Systems
- Used a siamese neural network model for face recognition, as it works most optimally in situations without huge amounts of data as well as small amounts of anchor images.
- face_detection.py contains functions for use in the facial recognition process and the actual model itself is placed in the facial_recognition.py, something to note is that, the import model command does not work on models with lamda layers in them, so in this case I built the entire model back manually and imported the weights from the model that was trained.
- The dataset that was used is a custom dataset personally crafted, filled with 40 people with 9-10 unique facial images for each person, totalling up to 378 images being used to train the facial recongition model.
- The model takes in an image and isolates the face through the facial detection model. A comparison is made between the faces in the image and previus taken face shots of people, previously onboarding images are compared to the isolated face and the image that is closest in simialrity is determined to be the identifed person. If the dissimialrity score is abvoe 0.8 the model will return unidentified.

>Other features
- Student creation form and process (Onbaording)
- the onbording process basically requires the user to place thier face in the view of the camera and through a facial detection model, the model would isolate the face and place that into an facial shot image that would be used later to identify people in images.
- attendance taking process

>Things to note
- Fixed the attendance error the occured during final presentation, students attendance will now properly upadte after every attendance taking.s
- The current processed_class_img.jpg shows a working example of the model properly predicting the users in the class img provided.

## Max
> AI Features
### Sentiment Analysis on Student Reflection
- Utilized a pretrained BERT model to extract sentiment of Student's reflection as BERT can perform sentiment analysis by leveraging its contextual understanding of language to identify the sentiment expressed in a given text.
- Used "bert-based-case" to tokenize the Input Ids.
- Preprocessed the student's reflection by encoding the Input Ids and setting the value of the Attention Mask as the length of the Input Ids.
- Added padding sequences to both the Input Ids and Attention Mask and returning it to the model using .predict().
- Prediction is the converted to a ndarray of ints using np.argmax() and if the output is 1 means that the reflection is predicted as positive and if the output is 2 means that it is negative.
- Dataset used: https://github.com/pg815/Student_Feedback_System/tree/main/dataset, it comes with labeled data (positive & negative)

> Other features
- Admin & Student Dashboard
- Login for Student/Teacher/Admin (includes validation)
- Creation of Student and Teacher via Admin dashboard (includes hashing and salting passwords)
- Create Topics via Admin dashboard
- Sentiment Graph and Table (in teacher dashboard)
- Student Reflection submission (includes: if student submited a reflection before, they will not be able to submit again)
- 404 and 500 Error handling
- Generation of Topics based on specific subjects (shown in student dashboard, each topic has its own reflection)