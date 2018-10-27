dataset=dataset[3:5]
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.6)
train=subset(dataset,split==T)
test=subset(dataset,split==F)

train[-3]=scale(train[-3])
test[-3]=scale(test[-3])
library(class)
y_pred=knn(train[-3],test[-3],cl=train$Purchased,k=5)
cm=table(test$Purchased,y_pred)
print(cm)
