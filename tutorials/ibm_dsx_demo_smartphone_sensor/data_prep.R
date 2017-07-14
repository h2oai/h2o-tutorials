# Reformat dataset
library(data.table)

d_train <- fread("./data/train.csv", data.table = F)
d_test <- fread("./data/test.csv", data.table = F)
features <- fread("./data/features.txt", data.table = F)

# Remove special char from features
features$V2 <- gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", features$V2)

# Rename columns
colnames(d_train) <- c("activity", paste0("f", 1:561, "_", features$V2))
colnames(d_test) <- c("activity", paste0("f", 1:561, "_", features$V2))

# Write to disk
fwrite(d_train, file = "./data/train.csv")
fwrite(d_test, file = "./data/test.csv")

system(paste("gzip -9 -v", "./data/train.csv"))
system(paste("gzip -9 -v", "./data/test.csv"))
