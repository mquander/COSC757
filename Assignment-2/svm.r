install.packages('e1071', dependencies = TRUE)
library(e1071)
library(MASS)

adult <- read.csv(file="UCI-dataset/adult.data", stringsAsFactors=TRUE)
adult_test <- read.csv(file="UCI-dataset/adult.test2.txt", stringsAsFactors=TRUE)
head(adult_test)
adult_copy <- adult

adult_gender_clean <- adult_copy
#adult_gender_clean$sex <-  gsub('Male', 1, adult_gender_clean$sex]

#adult_gender_clean

# get the numeric values of the adult dataset
adult_numerical_columns <- adult[, c(1, 5, 11:13, 15)]
adult_test_numerical_columns <- adult_test[, c(1, 5, 11:13, 15)]

# train svm using svm() in e1071 package
model <- svm(more.less.50k ~ ., adult_numerical_columns, type="C")
#2
model2 <- svm(more.less.50k ~ ., adult_numerical_columns, type="C")

print(model)
summary(model)

#2
print(model2)
summary(model2)

# plot the svm data (model)
plot(model, adult_numerical_columns, age~education.num)
plot(model, adult_numerical_columns, age~hours.per.week)

#2 - edunum better plot
plot(model2, adult, age~education.num)
plot(model2, adult, age~hours.per.week)

# prediction
prediction <- predict(model, adult_test_numerical_columns[, -6])
#2
prediction2 <- predict(model2, adult_test[, -15])

# confusion matrix
tab <- table(pred=prediction, true=adult_test_numerical_columns[, 6])
tab

#2
tab2 <- table(pred=prediction2, true=adult_test[, 15])
tab2

#sensitivity, the specificity and the precision 
classAgreement(tab)

#2
classAgreement(tab2)

# trim training set for tune()
trim_adult <- adult_copy[1:100, ]
trim_adult
tuned <- tune.svm(more.less.50k ~ ., data=trim_adult, gamma=10^(-6:-1), cost=10^(1:2))
summary(tuned)

#2
tuned2 <- tune.svm(more.less.50k ~ ., data=adult, gamma=10^(-6:-1), cost=10^(1:2))
summary(tuned2)

