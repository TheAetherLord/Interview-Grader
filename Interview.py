#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

class AIScreeningSystem:
    def __init__(self, criteria, questions):
        self.criteria = criteria
        self.questions = questions
        self.vectorizer = TfidfVectorizer()
        
    def evaluate_candidate(self, responses):
        scores = {}
        for criterion, ideal_response in self.criteria.items():
            response = responses.get(criterion, "")
            tfidf_matrix = self.vectorizer.fit_transform([ideal_response, response])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores[criterion] = similarity
        return scores

# Define criteria and ideal responses
criteria = {
    "machine_learning": "Machine learning is a subset of AI that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. It involves training models on data to make predictions or decisions without being explicitly programmed. Key concepts include supervised learning, unsupervised learning, and reinforcement learning. Machine learning algorithms can handle complex patterns and make data-driven decisions, making them valuable in various fields such as finance, healthcare, and technology.",
    "deep_learning": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. It's particularly effective for tasks like image and speech recognition, natural language processing, and other complex pattern recognition problems. Deep learning models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), can automatically learn hierarchical representations of data, enabling them to capture intricate patterns and relationships. This makes them powerful tools for solving complex problems in computer vision, natural language processing, and other domains.",
    "ethics_in_ai": "Ethics in AI involves considering the moral implications and societal impact of artificial intelligence systems. This includes issues such as bias in AI algorithms, privacy concerns, transparency in decision-making processes, and the potential impact of AI on employment and society at large. Key ethical principles in AI include fairness, accountability, transparency, and privacy. It's crucial to consider the potential unintended consequences of AI systems and to develop guidelines and regulations to ensure that AI is developed and deployed in a way that benefits society as a whole.",
    "programming_skills": "Strong programming skills in languages such as Python, R, or Java are essential for AI roles. Familiarity with libraries and frameworks like TensorFlow, PyTorch, or scikit-learn is also important. Additionally, understanding of data structures, algorithms, and software engineering principles is crucial. This includes proficiency in version control systems like Git, experience with database systems, and the ability to write efficient, scalable, and maintainable code. Knowledge of cloud computing platforms and big data technologies is also valuable in many AI roles.",
    "data_preprocessing": "Data preprocessing is a critical step in the machine learning pipeline. It involves cleaning, normalizing, and transforming raw data into a format suitable for analysis and model training. Key techniques include handling missing values, encoding categorical variables, scaling numerical features, and dealing with outliers. Effective data preprocessing can significantly improve model performance and reliability.",
    "model_evaluation": "Model evaluation is the process of assessing the performance and effectiveness of a machine learning model. It involves using various metrics and techniques to measure how well a model generalizes to unseen data. Common evaluation methods include cross-validation, confusion matrices, ROC curves, and metrics such as accuracy, precision, recall, and F1-score. Understanding model evaluation is crucial for selecting the best model and tuning hyperparameters."
}

questions = {
    "machine_learning": [
        "Can you explain what machine learning is and how it relates to AI?",
        "What are the main types of machine learning? Provide examples of use cases for each.",
        "Explain the difference between supervised and unsupervised learning.",
        "What is the role of training data in machine learning, and why is data quality important?"
    ],
    "deep_learning": [
        "What is deep learning, and how does it differ from traditional machine learning?",
        "Explain the concept of neural networks and how they work.",
        "What are some popular deep learning architectures, and what are they typically used for?",
        "How does deep learning handle feature extraction differently from traditional machine learning methods?"
    ],
    "ethics_in_ai": [
        "What are some key ethical considerations in the development and deployment of AI systems?",
        "How can bias be introduced into AI systems, and what strategies can be used to mitigate it?",
        "Discuss the importance of transparency and explainability in AI systems.",
        "What potential societal impacts of AI do you think are most important to consider, and why?"
    ],
    "programming_skills": [
        "Describe your programming experience and familiarity with AI/ML libraries and frameworks.",
        "How do you approach optimizing code for performance in machine learning applications?",
        "Explain your experience with version control systems and collaborative coding practices.",
        "Describe a challenging programming problem you've solved and how you approached it."
    ],
    "data_preprocessing": [
        "Why is data preprocessing important in machine learning projects?",
        "Describe some common techniques for handling missing data.",
        "How do you approach feature scaling, and why is it important?",
        "Explain the process of encoding categorical variables and when you might use different encoding methods."
    ],
    "model_evaluation": [
        "What metrics would you use to evaluate a classification model, and why?",
        "Explain the concept of cross-validation and why it's important.",
        "How do you interpret a ROC curve and AUC score?",
        "Describe how you would handle class imbalance in a dataset and how it affects model evaluation."
    ]
}

# Initialize the screening system
screening_system = AIScreeningSystem(criteria, questions)

# HTML template for the interview page
INTERVIEW_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Role Interview</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        h1, h2 { color: #333; }
        form { max-width: 800px; margin: 0 auto; }
        .question-group { margin-bottom: 30px; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        textarea { width: 100%; height: 100px; margin-top: 5px; }
        input[type="submit"] { display: block; margin-top: 20px; padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>AI Role Interview</h1>
    <form action="/submit" method="post">
        {% for criterion, question_list in questions.items() %}
        <div class="question-group">
            <h2>{{ criterion.replace('_', ' ').title() }}</h2>
            {% for question in question_list %}
            <label for="{{ criterion }}_{{ loop.index }}">{{ question }}</label>
            <textarea name="{{ criterion }}_{{ loop.index }}" id="{{ criterion }}_{{ loop.index }}" required></textarea>
            {% endfor %}
        </div>
        {% endfor %}
        <input type="submit" value="Submit Interview">
    </form>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interview Results</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        h1 { color: #333; }
        .result { margin-bottom: 20px; }
        .score { font-weight: bold; }
    </style>
</head>
<body>
    <h1>Interview Results</h1>
    <div class="result">
        <p>Overall Score: <span class="score">{{ "{:.2f}".format(overall_score) }}</span></p>
    </div>
    {% for criterion, score in scores.items() %}
    <div class="result">
        <h2>{{ criterion.replace('_', ' ').title() }}</h2>
        <p>Score: <span class="score">{{ "{:.2f}".format(score) }}</span></p>
    </div>
    {% endfor %}
</body>
</html>
"""

@app.route('/')
def interview():
    return render_template_string(INTERVIEW_TEMPLATE, questions=screening_system.questions)

@app.route('/submit', methods=['POST'])
def submit():
    responses = request.form.to_dict()
    
    # Combine responses for each criterion
    combined_responses = {}
    for key, value in responses.items():
        criterion = key.rsplit('_', 1)[0]
        if criterion not in combined_responses:
            combined_responses[criterion] = ""
        combined_responses[criterion] += value + " "

    scores = screening_system.evaluate_candidate(combined_responses)
    overall_score = np.mean(list(scores.values()))
    return render_template_string(RESULTS_TEMPLATE, overall_score=overall_score, scores=scores)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




