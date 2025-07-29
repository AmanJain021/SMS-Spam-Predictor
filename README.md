# MACHINE-LEARNING-MODEL-IMPLEMENTATION-FOR-SMS-SPAM-PREDICTION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: AMAN CHHAJED

*INTERN ID*: CT06DG2147

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

This project demonstrates a complete machine learning model pipeline using Python and scikit-learn to perform predictive analysis. The core objective of this task is to create a classification model capable of identifying whether a given message is spam or not, based on the textual content. The dataset used for training and evaluation is a labeled SMS spam collection, which contains a variety of real-world text messages marked as either "ham" (not spam) or "spam." This kind of classification is highly relevant in the domain of natural language processing (NLP), particularly for automating the detection and filtering of unsolicited messages, phishing content, or irrelevant communication in email and messaging platforms.The tools and technologies used in this project include Python (v3.13+), the scikit-learn machine learning library for model building and evaluation, pandas for data manipulation, and matplotlib for visualization. The model implementation was developed and tested using Jupyter Notebook, which provides an interactive coding environment ideal for step-by-step analysis and visualization of results. Additionally, the project uses other essential Python libraries such as NumPy and Seaborn to enhance the data exploration and charting processes.The process begins by loading and exploring the dataset. Text messages are preprocessed using standard NLP techniques, including case normalization, punctuation removal, tokenization, and vectorization. For transforming textual data into numerical features suitable for model training, the CountVectorizer from scikit-learn is utilized. This converts the collection of messages into a matrix of token counts, which is a common approach in bag-of-words modeling. The dataset is then split into training and testing sets using an 80:20 ratio to ensure fair evaluation.A Naive Bayes classifier — specifically the Multinomial Naive Bayes algorithm — is selected due to its effectiveness in text classification problems. The model is trained on the training set and evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score. These metrics provide a comprehensive view of how well the model is able to distinguish between spam and non-spam messages. The confusion matrix is also plotted to give a visual representation of the model’s classification performance.The project deliverables include the full Jupyter Notebook (SpamDetection.ipynb), which contains all code cells, explanations, charts, and results. This notebook not only showcases the end-to-end machine learning pipeline but also serves as a learning tool for beginners in NLP and classification modeling. The code is modular and well-commented, allowing for easy extension to other text classification tasks such as sentiment analysis or fake news detection.In practical terms, the approach implemented in this task can be used to build the foundation for intelligent filtering systems in email clients, messaging apps, and even customer support chatbots. By adapting the preprocessing pipeline and training on domain-specific datasets, similar models can also be applied in other industries like finance, healthcare, and e-commerce for classifying user feedback, detecting fraud, or analyzing trends.Overall, this project demonstrates not only the theoretical understanding of supervised learning but also the practical skills required to build, evaluate, and interpret a real-world classification model using industry-standard tools.

#  OUTPUT

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/17c23c37-7ebf-4790-9572-e64b3ba68787" />

<img width="539" height="455" alt="Image" src="https://github.com/user-attachments/assets/1f65bbcb-aedc-41d5-bf24-2dafb10ea10c" />
