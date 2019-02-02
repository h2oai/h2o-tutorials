
###################### Predicting Good Amazon Reviews ######################
# Amazon Fine Food Reviews dataset consists of 568,454 food reviews Amazon users left up to October 2012
# https://www.kaggle.com/snap/amazon-fine-food-reviews

# script is based off of the Craigslist Word2Vec Demo
# https://github.com/h2oai/h2o-3/blob/master/h2o-py/demos/word2vec_craigslistjobtitles.ipynb


###### Step 1 (of 10). Import data into H2O ###### 

# Start H2O Cluster
library('h2o')
h2o.init(max_mem_size = '8G', bind_to_localhost = FALSE)

# Import data
# https://s3-us-west-2.amazonaws.com/h2o-tutorials/data/topics/nlp/amazon_reviews/AmazonReviews.csv
reviews = h2o.importFile("data/topics/nlp/amazon_reviews/AmazonReviews.csv")

###### Step 2 (of 10). Exploratory Analysis ###### 

# Get dimension of the reviews data
dim(reviews)

# View head of the data
View(head(reviews))

# See how are the scores distributed
h2o.table(reviews$Score)

# View the most frequent review summaries
summary_freq <- h2o.table(reviews$Summary)
summary_freq <- h2o.arrange(summary_freq, desc(Count))
head(summary_freq)

# Add Target Column: "PositiveReview"
reviews$PositiveReview <- h2o.ifelse(reviews$Score >= 4, "1", "0")

h2o.table(reviews$PositiveReview)

###### Step 3 (of 10). Tokenize Words ###### 

# Set Stop Words
STOP_WORDS <- read.csv("/home/h2o/data/topics/nlp/amazon_reviews/stopwords.csv", stringsAsFactors = FALSE)$STOP_WORD
head(STOP_WORDS)

# Create a tokenize function that tokenizes the sentences and filters certain words
tokenize <- function(sentences, stop_word = STOP_WORDS){
  # Tokenize sentences by word
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  # Convert words to lowercase
  tokenized_lower <- h2o.tolower(tokenized)
  # Remove words with one letter
  tokenized_filter <- tokenized_lower[(h2o.nchar(tokenized_lower) >= 2) | is.na(tokenized_lower), ]
  # Remove words with any numbers
  tokenized_words <- tokenized_filter[h2o.grep(pattern = "[0-9]", x = tokenized_filter, invert = TRUE, output.logical = TRUE), ]
  # Remove stop words
  tokenized_words <- tokenized_words[is.na(tokenized_words) | !(tokenized_words %in% STOP_WORDS), ]
  
  return(tokenized_words)
}

# Break reviews into sequence of words
words <- tokenize(reviews$Text)

head(words)

###### Step 4 (of 10). Train Word2Vec Model ###### 

# Train Word2Vec Model for vec size = 2
w2v_len2_model <- h2o.word2vec(training_frame = words, vec_size = 2, model_id = "w2v_len2.hex")

###### Step 5 (of 10). Analyze Word Embeddings ######

# View the first 6 word embeddings
sample_embeddings <- words[c(1:6), ]
colnames(sample_embeddings) <- "Word"
sample_embeddings <- h2o.cbind(sample_embeddings, h2o.transform(w2v_len2_model, sample_embeddings, aggregate_method = "None"))

head(sample_embeddings)

# Get Word Embeddings for each unique word
word_embeddings <- h2o.toFrame(w2v_len2_model)

# Filter Word Embeddings to selected words
selected_words <- c("coffee", "espresso", "starbucks", "sweet", "salty", "savory", "email", "support", "answered", 
                    "unhappy", "waited", "returned", "tasty", "yummy", "moldy", "expired", "salmonella", "best", 
                    "amazing", "abdominal", "folic", "zinc")

filtered_embeddings <- word_embeddings[word_embeddings$Word %in% selected_words, ]
plot_data <- as.data.frame(filtered_embeddings)

# Plot Word Embeddings
library('ggplot2')
ggplot(plot_data, aes(x=V1, y=V2, label=Word)) + geom_text(check_overlap = TRUE) + theme_bw()

# Train Word2Vec Model for vec size = 100
w2v_model <- h2o.word2vec(training_frame = words, vec_size = 100, model_id = "w2v.hex")

# Sanity check - find synonyms for the word 'coffee'
h2o.findSynonyms(w2v_model, "coffee", count = 5)

# Sanity check - find synonyms for the word 'stale'
h2o.findSynonyms(w2v_model, "stale", count = 5)

# Cluster Word Embeddings
word_embeddings <- h2o.toFrame(w2v_model)

# Train K-Means Model
kmeans <- h2o.kmeans(x = setdiff(colnames(word_embeddings), c("Word", "Count")),
                     training_frame = word_embeddings,
                     model_id = "word_segments.hex",
                     estimate_k = TRUE, k = 100, # Max number of clusters
                     seed = 1234)

h2o.centroid_stats(kmeans)

word_clusters <- h2o.cbind(word_embeddings, h2o.predict(kmeans, word_embeddings))
head(word_clusters[c("Word", "predict")])

selected_clusters <- word_clusters[word_clusters$Word %in% selected_words, ]

cluster_ids <- as.matrix(h2o.table(selected_clusters$predict)$predict)[, 1]
for(id in cluster_ids){
  print(paste0("Cluster ", id))
  print(selected_clusters[selected_clusters$predict == id, "Word"])
}

###### Step 6 (of 10). Train Model with Original Data ######

# Find the 80th quantile of time in the dataset
time_split <- h2o.quantile(reviews$Time, prob = 0.8)
reviews$Train <- h2o.ifelse(reviews$Time < time_split, "Yes", "No")

train <- reviews[reviews$Train == "Yes", ]
test <- reviews[reviews$Train == "No", ]

# Train GBM Model
predictors <- c('ProductId', 'UserId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time')
response <- 'PositiveReview'

gbm_baseline <- h2o.gbm(x = predictors, y = response,
                        training_frame = train, validation_frame = test,
                        stopping_metric = "AUC", stopping_tolerance = 0.001, 
                        stopping_rounds = 5, score_tree_interval = 10, 
                        model_id = "gbm_baseline.hex"
                        )

# Get Performance Metrics
print(paste0("AUC on Validation Data: ", round(h2o.auc(gbm_baseline, valid = TRUE), 3)))
h2o.confusionMatrix(gbm_baseline, valid = TRUE)

# Plot Variable Importance
h2o.varimp_plot(gbm_baseline)

# View Partial Dependency Plot for HelpfulnessNumerator
pdp_helpfulness <- h2o.partialPlot(gbm_baseline, train, cols = c("HelpfulnessNumerator"))

###### Step 7 (of 10). Train Model with Word Embeddings ######

# Calculate a vector for each review
review_vecs <- h2o.transform(w2v_model, words, aggregate_method = "AVERAGE")
head(review_vecs)

# Add aggregated word embeddings 
ext_reviews <- h2o.cbind(reviews, review_vecs)

# Split data into training and testing
ext_train <- ext_reviews[ext_reviews$Train == "Yes", ]
ext_test <- ext_reviews[ext_reviews$Train == "No", ]

# Train new GBM model
predictors <- c(predictors, colnames(review_vecs))
response <- 'PositiveReview'

gbm_embeddings <- h2o.gbm(x = predictors, y = response, 
                          training_frame = ext_train, validation_frame = ext_test,
                          stopping_metric = "AUC", stopping_tolerance = 0.001,
                          stopping_rounds = 5, score_tree_interval = 10,
                          ntrees = 1000,
                          model_id = "gbm_embeddings.hex"
)

# Compare Performance
print(paste0("Baseline AUC: ", round(h2o.auc(gbm_baseline, valid = TRUE), 3)))
print(paste0("With Embeddings AUC: ", round(h2o.auc(gbm_embeddings, valid = TRUE), 3)))

h2o.confusionMatrix(gbm_embeddings, valid = TRUE)

# Plot Variable Importance
h2o.varimp_plot(gbm_embeddings)

# Train a simpler GLM model using important word2vec features of the GBM model to generate interactions
features_by_importance <- h2o.varimp(gbm_embeddings)$variable
top_w2v_features <- features_by_importance[grepl("^C", features_by_importance)][1:10]
print(top_w2v_features)

glm_predictors <- c("HelpfulnessNumerator", "HelpfulnessDenominator", colnames(review_vecs))

glm_embeddings <- h2o.glm(x = glm_predictors, y = response, 
                          training_frame = ext_train, validation_frame = ext_test,
                          interactions = top_w2v_features, family = "binomial")

# Compare Performance
print(paste0("Baseline AUC: ", round(h2o.auc(gbm_baseline, valid = TRUE), 3)))
print(paste0("With Embeddings AUC (GBM): ", round(h2o.auc(gbm_embeddings, valid = TRUE), 3)))
print(paste0("With Embeddings AUC (GLM): ", round(h2o.auc(glm_embeddings, valid = TRUE), 3)))


h2o.confusionMatrix(glm_embeddings, valid = TRUE)

# Plot Variable Importance
h2o.varimp_plot(glm_embeddings, 10)

###### Step 8 (of 10). Run AutoML ######
automl <- h2o.automl(x = predictors, y = response,
                     training_frame = ext_train, leaderboard_frame = ext_test,
                     project_name = "positive_reviews", max_runtime_secs = 180, 
                     keep_cross_validation_models = FALSE, keep_cross_validation_predictions = FALSE,
                     nfolds = 3, exclude_algos = c("DRF"), seed = 1234)

automl@leaderboard

###### Step 9 (of 10). Watch AutoML progress (in the H2O Flow Web UI) ######

# Go to port 54321
# In H2O Flow, go to Admin -> Jobs
# Click on the "Auto Model" job with the "positive_reviews" job name and explore it

###### Step 10 (of 10). Shutdown the H2O Cluster ######
h2o.shutdown()
