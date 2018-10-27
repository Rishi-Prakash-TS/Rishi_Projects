dataset=dataset[3:5]
dataset$Purchased=factor(dataset$Purchased,levels = c(0,1))
#IN ORDER TO CHANGE THE OUTPUT COLUMN FROM NUMBERS TO FACTORS

library(caTools)
set.seed(123)

split=sample.split(dataset$Purchased,SplitRatio = 0.6)
train_set=subset(dataset,split==T)
test_set=subset(dataset,split==F)

train_set[-3]=scale(train_set[-3])
test_set[-3]=scale(test_set[-3])

library(e1071)

classifier=naiveBayes(train_set[-3],train_set$Purchased)

y_pred=predict(classifier,test_set[-3])
table(test_set$Purchased,y_pred)
