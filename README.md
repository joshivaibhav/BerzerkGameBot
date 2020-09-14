# Exploring Discourse in Reddit

Our goal here is to build a machine learning agent that can predict which subreddit an unlabeled post comes from. We aim to acheive this via implementing various machine learning algorithms as well as deep learning techniques. We then compare and contrast the various algorithms and deduce which is the best for the job.


# Data
Each data record is naturally a reddit post. We choose any two subreddits as our target classes for the posts. We split the data records at around 1000 
for each target class.
####
The posts are fetched using Reddit's PRAW API.

# Classifiers

  * Naive Bayes
  * SVM
  * Random Forest
  * Logistic
# Neural Networks
  * LSTM (Long Short-Term Memory)

# Report and Analysis
  * We compare accuracies and generate ROC Curves and Error curves for a deeper analysis
  * Feel free to dive into the Report pdf for more details.
  
