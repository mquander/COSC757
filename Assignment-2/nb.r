install.packages('e1071', dependencies = TRUE)
library(class)
library(e1071)

adult <- read.csv(file="UCI-dataset/adult.data", stringsAsFactors=TRUE)
adult_test <- read.csv(file="UCI-dataset/adult.test2.txt", stringsAsFactors=TRUE)

#adult_copy <- adult
adult_numerical_columns <- adult[, c(1, 5, 11:13, 15)]
adult_test_numerical_columns <- adult_test[, c(1, 5, 11:13, 15)]

pairs(adult_numerical_columns, main="Adult Data Set", pch=21, bg=c("red", "green")[unclass(adult_numerical_columns$more.less.50k)])

plot(pairs)

summary(adult_numerical_columns)
adult_numerical_columns$more.less.50k <- factor(adult_numerical_columns$more.less.50k)


# naive bayes model and confusion matrix
classifier <- naiveBayes(adult_numerical_columns[,1:5], adult_numerical_columns[,6])
table(predict(classifier, adult_numerical_columns[,-6]), adult_numerical_columns[,6])

#classifier <- naiveBayes(adult_test_numerical_columns[,1:5], adult[,6])
tab <- table(predict(classifier, adult_test_numerical_columns[,-6]), adult_test_numerical_columns[,6])
tab
head(adult_test_numerical_columns)
head(adult_numerical_columns)

summary(adult)
summary(adult_test)



# does it work on non-numeric data?
classifier_testonallcols <- naiveBayes(adult[,1:15], adult[,15])
tab_testonallcols <- table(predict(classifier, adult[,-15]), adult[,15])
tab_testonallcols






# gender proportion test set
male_count_test <- str_count(adult_test$sex,'Male') 
male_count_test
prop_male_test <- sum(male_count_test == 1) / length(adult_test$sex)
prop_male_test
#private industry proportion test set
private_count_test <- str_count(adult_test$workclass, 'Private')
private_count_test
prop_private_test <- sum(private_count_test == 1) / length(adult_test$workclass)
prop_private_test
#native US proportion test set
US_count_test <- str_count(adult_test$native.country, 'United-States')
US_count_test
prop_US_test <- sum(US_count_test == 1) / length(adult_test$native.country)
prop_US_test

# sd for training numerical variables
sd_age_training <- sd(adult$age)
sd_age_training

# edunum
sd_edunum_training <- sd(adult$education.num)
sd_edunum_training

#capgain
sd_capgain_training <- sd(adult$capital.gain)
sd_capgain_training

#caploss
sd_caploss_training <- sd(adult$capital.loss)
sd_caploss_training

#hrswk
sd_hrswk_training <- sd(adult$hours.per.week)
sd_hrswk_training

#sd for test numerical variables
#age
sd_age_test <- sd(adult_test$age)
sd_age_test

#edunum
sd_edunum_test <- sd(adult_test$education.num)
sd_edunum_test

#capgain
sd_capgain_test <- sd(adult_test$capital.gain)
sd_capgain_test

#caploss
sd_caploss_test <- sd(adult_test$capital.loss)
sd_caploss_test

#hrswk
sd_hrswk_test <- sd(adult_test$hours.per.week)
sd_hrswk_test

age_tdata <- (38.58-38.77)/sqrt(((13.64043^2)/32561) + ((13.84919^2)/16281))
pvalue_age <- 2*pt(age_tdata, df=16280, lower.tail=FALSE)
age_tdata; pvalue_age

edunum_tdata <- (10.08-10.07)/sqrt(((2.57272^2)/32561) + ((2.567545^2)/16281))
pvalue_edunum <- 2*pt(edunum_tdata, df=16280, lower.tail=FALSE)
edunum_tdata; pvalue_edunum

capgain_tdata <- (1078-1082)/sqrt(((7385.292^2)/32561) + ((7583.936^2)/16281))
pvalue_capgain <- 2*pt(capgain_tdata, df=16280, lower.tail=FALSE)
capgain_tdata; pvalue_capgain

caploss_tdata <- (87.3-87.9)/sqrt(((402.9602^2)/32561) + ((403.1053^2)/16281))
pvalue_caploss <- 2*pt(caploss_tdata, df=16280, lower.tail=FALSE)
caploss_tdata; pvalue_caploss

hrswk_tdata <- (40.44-40.39)/sqrt(((12.34743^2)/32561) + ((12.47933^2)/16281))
pvalue_hrswk <- 2*pt(hrswk_tdata, df=16280, lower.tail=FALSE)
hrswk_tdata; pvalue_hrswk
