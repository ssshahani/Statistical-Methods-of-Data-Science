# this is a seperate file to divide the whole data set to training and testing sets.
# seperating this is due to the fact that we need to fix the testing set for all methods. 
setwd("F:/MA543/FinalProjectCode/")
voice_data = read.csv("voice.csv")

# first dividing (fix this part of code)
set.seed(1)
whole_index = 1:dim(voice_data)[1]
train_index = sample(dim(voice_data)[1], dim(voice_data)[1]*0.75)
test_index = whole_index[-train_index]

train_set <- function(){
  return (voice_data[train_index, ])
}
test_set <- function(){
  return (voice_data[test_index, ])
}
