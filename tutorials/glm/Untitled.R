cut_column <- function(data, col) {
  min_val = min(data$Train[,col])-1
  max_val = max(data$Train[,col])+1
  x = h2o.hist(data$Train[, col])
  breaks = x$breaks[which(x$counts > 1000)]
  lvls = c("min",paste("i_",breaks[2:length(breaks)-1],sep=""),"max")
  col_cut <- paste(col,"_cut",sep="")
  data$Train[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Train[,col],breaks=c(min_val,breaks,max_val)),lvls)
  if(!is.null(data$Test)) {
    min_val = min(data$Test[,col])-1
    max_val = max(data$Test[,col])+1
    data$Test[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Test[,col],breaks=c(min_val,breaks,max_val)),lvls)
  }
  if(!is.null(data$Valid)) {
    min_val = min(data$Valid[,col])-1
    max_val = max(data$Valid[,col])+1
    data$Valid[,col_cut] <- h2o.setLevels(h2o.cut(x = data$Valid[,col],breaks=c(min_val,breaks,max_val)),lvls)
  }
  data
}

interactions <- function(data, cols, pairwise = TRUE) {
  iii = h2o.interaction(data = data$Train, destination_frame = "itrain",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
  data$Train <- h2o.cbind(data$Train,iii)
  if(!is.null(data$Test)) {
    iii = h2o.interaction(data = data$Test, destination_frame = "itrain",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
    data$Test <- h2o.cbind(data$Test,iii)
  }
  if(!is.null(data$Valid)) {
    iii = h2o.interaction(data = data$Valid, destination_frame = "itrain",factors = cols,pairwise=pairwise,max_factors=1000,min_occurrence=100)
    data$Valid <- h2o.cbind(data$Valid,iii)
  }
  data
}

add_features <- function(data) {
  names(data) <- c("Train","Test","Valid")
  data = cut_column(data,'Elevation')
  data = cut_column(data,'Hillshade_Noon')
  data = cut_column(data,'Hillshade_9am')
  data = cut_column(data,'Hillshade_3pm')
  data = cut_column(data,'Horizontal_Distance_To_Hydrology')
  data = cut_column(data,'Slope')
  data = cut_column(data,'Horizontal_Distance_To_Roadways')
  data = cut_column(data,'Aspect')
  interaction_cols = c("Elevation_cut","Wilderness_Area","Soil_Type","Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut","Horizontal_Distance_To_Hydrology_cut","Slope_cut","Horizontal_Distance_To_Roadways_cut","Aspect_cut")
  data = interactions(data, interaction_cols)
  interaction_cols2 = c("Hillshade_Noon_cut","Hillshade_9am_cut","Hillshade_3pm_cut")
  data = interactions(data, interaction_cols2,pairwise = FALSE)
  data
}

h2o.removeAll()
setwd('/Users/tomasnykodym/h2o-world-2015-training/tutorials/')
D = h2o.importFile(paste(getwd(),'data/covtype.full.csv',sep="/"))
data = h2o.splitFrame(D,ratios=c(.7,.15),destination_frames = c("train","test","valid"))
names(data) <- c("Train","Test","Valid")
y = "Cover_Type"
x = names(data$Train)
x = x[-which(x==y)]
m1 = h2o.glm(training_frame = data$Train, validation_frame = data$Valid, x = x, y = y,family='multinomial',solver='L_BFGS',lambda=1e-4)
# m2 = h2o.glm(training_frame = data$Train, validation_frame = data$Valid, x = x, y = y,family='multinomial',solver='IRLSM',alpha=.99) # runs 2.5 mins on laptop


D_binomial = D[D$Cover_Type %in% c("class_1","class_2"),]
h2o.setLevels(D_binomial$Cover_Type,c("class_1","class_2"))
data_binomial = h2o.splitFrame(D_binomial,ratios=c(.7,.15),destination_frames = c("train_b","test_b","valid_b"))
names(data_binomial) <- c("Train","Test","Valid")
y = "Cover_Type"
x = names(data_binomial$Train)
x = x[-which(x==y)]
m_binomial = h2o.glm(training_frame = data_binomial$Train, validation_frame = data_binomial$Valid, x = x, y = y, family='binomial')

data_binomial_ext <- add_features(data_binomial)
data_binomial_ext$Train <- h2o.assign(data_binomial_ext$Train,"train_b_ext")
data_binomial_ext$Valid <- h2o.assign(data_binomial_ext$Valid,"valid_b_ext")
data_binomial_ext$Test <- h2o.assign(data_binomial_ext$Test,"test_b_ext")
y = "Cover_Type"
x = names(data_binomial_ext$Train)
x = x[-which(x==y)]
m_binomial_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial')
# does not run, we got too many columns -> try L-BFGS
m_binomial_2_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial',alpha=0, solver='L_BFGS')
# same as above, training same as test, maybe too much regularization? try again with lower lambda (alpha is 0 by default for L-BFGS)
m_binomial_3_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial',alpha=0, solver='L_BFGS', lambda=1e-4)
# way better, 17% error, down from 24%, but is it optimal? We can run lambda search
# let's run default alpha, IRLSM works better with L1 and we can use it now(lambda search + strong rules)! 
m_binomial_4_ext = h2o.glm(training_frame = data_binomial_ext$Train, validation_frame = data_binomial_ext$Valid, x = x, y = y, family='binomial',lambda_search=TRUE)
# similar to m3, slightly worse but at the edge of regularization, look slike we don't need much regulariztaion, might as well run the L2 only L-BFGS (faster)

# let's revisit the multinomial case with our new features
data_ext <- add_features(data)
data_ext$Train <- h2o.assign(dat_ext$Train,"train_m_ext")
data_ext$Valid <- h2o.assign(data_ext$Valid,"valid_m_ext")
data_ext$Test <- h2o.assign(data_ext$Test,"test_m_ext")
y = "Cover_Type"
x = names(data_ext$Train)
x = x[-which(x==y)]
m2 = h2o.glm(training_frame = data_ext$Train, validation_frame = data_ext$Valid, x = x, y = y,family='multinomial',solver='L_BFGS',lambda=1e-4)
# 21% err down from 28%
m2



