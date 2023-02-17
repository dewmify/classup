# classup
## Joshua

## Ryo

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