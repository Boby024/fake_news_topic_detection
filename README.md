# fake_news_topic_detection

To allow the classifier to be better in classification (fake news detection and Topical Domain Classification):
<ul>
  <li>Combining columns "headlne" and "content" from dataset and  exclude row without content if these exist</li>
  <li>data cleaning ( removing email address, hyperlinks, numbers, special characters and duplicate)</li>
  <li>
  After testing Decision Tree, Random Forest and Multinomial Naive Bayes (with parameter such as unigram, bigram), we decide to use Multinomial Naive Bayes algorithm (using Scikit-learn library to classify text cleaned) based on its result
  </li>
</ul>
