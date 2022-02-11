install.packages("ggplot2")
install.packages("stringr")
install.packages("plyr")
install.packages("rpart")
install.packages("rpart.plot")
library(ggplot2)
library(stringr)
library(plyr)
library(ROCR)
library(rpart)
library(rpart.plot)

adult <- read.csv(file="UCI-dataset/adult.data", stringsAsFactors=TRUE)
adult_test <- read.csv(file="UCI-dataset/adult.test2.txt", stringsAsFactors=TRUE)
# summary statistics for the continuous variables of focus
age_summary <- summary(adult$age)
age_summary

years_edu_summary <- summary(adult$education.num)
years_edu_summary

cap_gain_summary <- summary(adult$capital.gain)
cap_gain_summary

cap_loss_summary <- summary(adult$capital.loss)
cap_loss_summary

hours_week_summary <- summary(adult$hours.per.week)
hours_week_summary

# summary statistics for the non-continuous variables of focus
workclass_summary <- summary(adult$workclass)
workclass_summary

marital_status_summary <- summary(adult$marital.status)
marital_status_summary

occupation_summary <- summary(adult$occupation)
occupation_summary

race_sum <- summary(adult$race)
race_sum

sex_sum <- summary(adult$sex)
sex_sum

nat_country_summary <- summary(adult$native.country)
nat_country_summary

#histograms
hist(adult$age,
     #xlim = c(-.125, .125),
     main="Histogram of Age",
     ylab="Count",
     xlab="Age",
     breaks=100,
     col="blue")
hist(adult$education.num,
     #xlim = c(-.125, .125),
     main="Histogram of Years of Education",
     ylab="Count",
     xlab="Years",
     breaks=100,
     col="blue")


# plot of counts of all marital statuses
ggplot(adult) + aes(x=age, group=marital.status, fill=marital.status) + 
        geom_histogram(binwidth=1, color='blue')

# plot of workclass
ggplot(adult) + aes(x=age, group=workclass, fill=workclass) + 
        geom_histogram(binwidth=1, color='blue')

# to plot income, change condition to 0 or 1?
# plot of workclass
ggplot(adult) + aes(x=age, group=more.less.50k, fill=more.less.50k) + 
        geom_histogram(binwidth=1, color='blue')

# plot of education-num, income
ggplot(adult) + aes(x=education.num, group=more.less.50k, fill=more.less.50k) + 
        geom_histogram(binwidth=1, color='blue')

# plot of marital.status, income
ggplot(adult) + aes(x=marital.status, group=more.less.50k, fill=more.less.50k) + 
        geom_histogram(binwidth=1, color='blue')


# proportion earning grater than $50k
greater_50k_count <- str_count(adult$more.less.50k, '>50K')
greater_50k_count
prop_greater_50k <- sum(greater_50k_count == 1)/length(adult$more.less.50k)
prop_greater_50k

# gender proportion
male_count <- str_count(adult$sex,'Male') 
male_count
male_count_num <-  sum(male_count == 1)
male_count_num
prop_male <- sum(male_count == 1) / length(adult$sex)
prop_male
#private industry proportion
private_count <- str_count(adult$workclass, 'Private')
private_count
private_count_num <- sum(private_count == 1)
private_count_num
prop_private <- sum(private_count == 1) / length(adult$workclass)
prop_private
#native US proportion
US_count <- str_count(adult$native.country, 'United-States')
US_count
US_count_num <- sum(US_count == 1)
US_count_num
prop_US <- sum(US_count == 1) / length(adult$native.country)
prop_US

# gender proportion test set
male_count_test <- str_count(adult_test$sex,'Male') 
male_count_test
male_count_test_num <- sum(male_count_test == 1)
male_count_test_num
prop_male_test <- sum(male_count_test == 1) / length(adult_test$sex)
prop_male_test
#private industry proportion test set
private_count_test <- str_count(adult_test$workclass, 'Private')
private_count_test
private_count_test_num <- sum(private_count_test == 1)
private_count_test_num
prop_private_test <- sum(private_count_test == 1) / length(adult_test$workclass)
prop_private_test
#native US proportion test set
US_count_test <- str_count(adult_test$native.country, 'United-States')
US_count_test
US_count_test_num <- sum(US_count_test == 1)
US_count_test_num
prop_US_test <- sum(US_count_test == 1) / length(adult_test$native.country)
prop_US_test

ppooled_male <- (male_count_num +male_count_test_num)/(32561+16281)
zdata_male <- (prop_male - prop_male_test)/sqrt(ppooled_male*(1-ppooled_male)*((1/32561)+(1/16281)))
pvalue_male <- 2*pnorm(abs(zdata_male), lower.tail=FALSE)
pvalue_male

ppooled_private <- (private_count_num +private_count_test_num)/(32561+16281)
zdata_private <- (prop_private - prop_private_test)/sqrt(ppooled_private*(1-ppooled_private)*((1/32561)+(1/16281)))
pvalue_private <- 2*pnorm(abs(zdata_private), lower.tail=FALSE)
pvalue_private

ppooled_US <- (US_count_num +US_count_test_num)/(32561+16281)
zdata_US <- (prop_US - prop_US_test)/sqrt(ppooled_US*(1-ppooled_US)*((1/32561)+(1/16281)))
pvalue_US <- 2*pnorm(abs(zdata_US), lower.tail=FALSE)
pvalue_US


# Binning
#n <- length(adult$marital.status)
#whichbin <- c(rep(0, n))
#nbins <- 7

#k-means clustering
#data <- adult$marital.status
#kmeansclustering <- kmeans(data, centers=nbins)
#whichbin <- kmeansclustering$cluster
#whichbin

#binning by 3 arbitrary cuts and creating a frequency distribution
#whichbincuts = cut(whichbin, 7)
#whichbincuts.freq = table(whichbincuts)
#whichbincuts.freq

#Equal widthbinning by 5 arbitrary cuts and creating a frequency distribution
#tableData = cut(data, 5) #, labels = c("Low", "Med-Low", "Med", "Med-High", "High")
#tableData.freq = table(tableData)
#tableData.freq

testResult <- rpart(more.less.50k ~ age + workclass + education.num + hours.per.week, data = adult, method = "class", rpart.control(minsplit=10, cp=0.001))
testRpart <- rpart(more.less.50k ~ age + workclass + education.num + occupation + sex + hours.per.week, data = adult, method = "class")
rpart.plot(testRpart, type=4, extra=100)
summary(testRpart)
print(testRpart)

# for test data set
testSetRpart <- rpart(more.less.50k ~ age + workclass + education.num + occupation + sex + hours.per.week, data = adult_test, method = "class")
rpart.plot(testSetRpart, type=4, extra=100)
summary(testSetRpart)
print(testSetRpart)


