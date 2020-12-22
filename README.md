# reddit-comment-removal


# Predicting Comment Removal from r/worldnews: January 2019

### Table of Contents  
[Description](#description)  
[Data Preparation](#data-preparation)   
[Exploratory Data Analysis](#exploratory-data-analysis)<br/>
[Modeling](#modeling)
[Results](#results)
[Summary](#summary)

## Description

Reddit is a website that aggregates news and social content and provides a platform for discussion. Although basically any topic can be discussed on Reddit, the site is heavily moderated. Users report offensive and abusive comments or comments that otherwise violate subreddit rules to moderators, who then review these comments and decide whether or not to remove them. 

This project predicts whether or not comments from Reddit’s PoliticalDiscussion subreddit will be removed, in order to potentially reduce the amount of user and moderator work. In particular, I investigate whether or not textual features from the comments themselves are predictive of comment removal, using supervised learning models such as logistic regression, random forest, gradient boosting, and Naive-Bayes classification. 

## Data Preparation

My dataset comprises 826,163 total comments, of which 24,952 (3%) are removed by moderators. I discuss different techniques I use to address the class imbalance later on in the modeling phase. 

My data collection process involves first querying both removed/user-deleted and intact comments from r/worldnews for the month of January 2019 from Google BigQuery. In order to restore specifically moderator-removed comments (marked with "[removed]"), I next retrieved the body text for these comments through Reddit's pushshift.api. 

I then merged these comments with intact ones and removed unnecessary columns. Next I removed automoderator-flagged comments (comments that generated an automatic response or were automatically flagged as spam: these comments did not contain the original textual context), user-deleted (marked by "[deleted]" in the text body), and otherwise missing (removed comments that could not be restored) comments. After this, I then created my target variable as an indicator showing whether or not a comment has been removed (1 = removed, 0 = intact). 

My text featurization pipeline is as follows:
1) Normalize text to convert comments into more uniform sequences
2) Remove punctuation, unnecessary characters, and stopwords
3) Lowercase and lemmatize words
4) Create bag of words and term frequency-inverse document frequency (tf-idf) matrices.

I noticed that there were a few removed comments using just the words "buh bye" repeated a few thousand times. I removed these comments so as to not bias my modeling results. 

## Exploratory Data Analysis

A few examples of removed comments are below:

"Lmao did I trigger you soy boy? Go back to shaving with Gillette”

“You people are deranged”

“Not an argument.”

“Perhaps finally some justice.”

The first two comments above can be easily construed as insulting. The last two, at least taken out of context, don't seem directly offensive, but if they could be if we do consider the context.

In terms of word importance, intact and removed comments do share some words in common, such as "think", "know", and "country". However, removed comments feature more profanity, more mentions of "Trump", and more prejudice-oriented words compared to intact comments.

<img src="imgs/intact_wordcloud.png" width = "450"/>  <img src="imgs/removed_wordcloud.png" width = "450"/>
