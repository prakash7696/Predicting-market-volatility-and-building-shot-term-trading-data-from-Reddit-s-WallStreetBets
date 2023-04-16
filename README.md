## Predicting-market-volatility-and-building-short-term-trading-strategies-using-data-from-Reddits-Wall

Aim - To make a **prediction** on stock prices and market volatility. predict the future price/and analyz the dynamics of the stock market of 500+ trading company.

## Understanding the Dataset

The dataset we are working on is a combination of YAHOO FINANCE and the S&P stock from 2013 to 2018.
The S&P stock contains the core stock market information  
The yahoo finance is real time dataset with a historical Data 
The Wall Street Json from Reddit on each day from 2012 to 2018.

## Output
target varible = 1 if increase the price of stock from pervious data a particular company

- The news dataset contain
contains open, high, low,close, volume, Adjclose

-S&P stock contains
each day such as Open, Close, High, Low,and Volume, Name.
and its also contain Open, Close, High, Low,and Volume for each 500+ company

date	         open           high     	low	          close       	volume             Name
          
2013-02-08                    15.07     15.12	14.63	14.75	8407500	        AAL

2013-02-11	          14.89	15.01	14.26	14.46	8882000	        AAL

2013-02-12	          14.45	14.51	14.10	14.27	8126000	        AAL

2013-02-13	          14.30	14.94	14.25	14.66	10259500        AAL

2013-02-14	          14.94	14.96	13.16	13.99	31879900        AAL

-The Wall Street Json from Reddit contain
 {"body","score_hidden","archived","name","author","downs","created_utc","subreddit_id","link_id","parent_id","score"}

## Text Preprocessing and Sentiment Analysis of wallstreetBets data
**Many columns in this data set but we need only "body","score" part
**Removing @ mentions, hashtags,the hyper-link but don't remove STOPWORD because of ticker
**We filled out the NaN values in the missed three topics.
** got the polarity, cont_sent,and cont_len for the Wall Street comments.
"Polarity" is of 'float' type and lies in the range of -1, 1, where 1 means a high positive sentiment, and -1 means a high negative sentiment. 
"cont_len" is the length of each comment. 
"cont_sent" Describes whether each comment has Positive,Negative, or Neutral sentiment. 
So, they will be very helpful in determining the increase or decrease of the stock market.


##Data Preprocessing on Stock Data
**Then we checked the missed values in the stock market information, it was complete. 
** remove duplicate 
** Standardizing the data helps to bring all the columns to the same scale

## Merge the both dataset
merge_df=pd.merge(SP,WSB_df, how='inner',on='date_parsed')
we merged the sentiment information (polarity and comment length) by date with the stock market information (Date,Open, High, Low, Close, Volume) in merged_data dataframe.
Before modelling and after splitting we scaled the data using standardization to shift the distribution to have a mean of zero and a standard deviation of one.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_valid)'''

## EDA 
**Introduction:** 
Now, final dataset contain this columns Date,Open,High,Low,Close,Volume,sent_pol,cont_sent

**Information of Dataset:**
Using countplot on target variable


**Univariate Analysis:**
plotted histogram to see the distribution of data for each column and found that few variables are normally distributed. 
However, we can't really say about that which variables needed to be studied. 
Subjectivity and polarity variable are derived ones and other historical stock variables required to sudy more that how they are related to each oyher.

**Descriptive Statistics:**
One-Hot Encoding On cont_sent
Parsing Data into day,year,month,date
Newdataset contain Open,High,Low,Close,Volume,Negative_score,Neutral_score,Positive_score,Compound_score,date_year,data_month,date_day,Target_Variable
NAN,Duplicate, any 

**Correlation Plot of Numerical Variables:**
All the continuous variables are positively correlated with each other with correlation coefficient but few are negatively correlated 
"Mutual Information Scores" is between all varible with score. close value is 0.025

**Visualisation of Variables:**
-Plot distribution graph between Count and Volume
- countGraph between positive, negative and Neutral with count. mostly are positive related
- Count Graph Target_Variable with count. ratio of 1 and 0 is 6/5.
- Pairplot visualizes for all data to find the relationship between them where the variables can be continuous or categorical
- BarPlot between Sentiment and Target_Variable.
- scatterplot between High vs Low Price it is almost increasing order 

## Final Dataset for modelling


#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

## Model Building
** LogisticRegression model and Naive Bayes 

**for Whole Dataset means contains all the company
X=  = ['date_parsed','open','high','low','close','volume','compound','neg','neu','pos','Negative','Neutral','Positive']
Y = 'Target_Variable'
* LogisticRegression model around 83% ACC

* NaiveBayes model around 50% acc

** For top 5 company use 5 particular company ticker
# scale the features
sc = StandardScaler()
X_scaled_nflx = sc.fit_transform(X_nflx[features_nflx])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

* LogisticRegression model
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
model = LogisticRegression()
model.fit(X_train, y_train)
GridSearchCV(model,param_grid=parameter,scoring='accuracy',cv=5)
LogisticRegression model accuracy(in %) for apple:0.7131474103585658
confusion matrix
[[ 95  29]
 [ 11 116]]

* Gaussian NaiveBayes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
Gaussian Naive Bayes model accuracy(in %) for apple: 49.00398406374502
confusion matrix 
[[108  18]
 [ 21 104]]

## Choosing the features
After choosing LDA model based on confusion matrix here where **choose the features** taking in consideration the deployment phase.

We know from the EDA that all the features are highly correlated and almost follows the same trend among the time.
So, along with polarity and subjectivity we choose the open price with the assumption that the user knows the open price but not the close price and wants to figure out if the stock price will increase or decrease.

When we apply the **logistic regression** model the accuracy dropped from 80% to 55%.





## Deployment 
###Django
-Model building

- Save model to joblib

- Create the app by Django startspp command

- Create the app using yahoo finance API
 
- Taking input from user Start date , end date, company name for the stock

- Showing the prediction ** Output ** with a graph for the given date














