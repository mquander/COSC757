# Istanbul Stock Exchange data

# import data into a dataframe
istanbul_ISE <- read.csv(file="Assignment-1/data_akbilgic_text.txt", stringsAsFactors=FALSE)

istanbul_ISE

# Summary of the ISE returns in USD
sum.istanbul_ISE <- summary(istanbul_ISE$USD.BASED.ISE)
sum.istanbul_ISE

# Summary of the ISE returns in USD
sum.istanbul_ISE_TL <- summary(istanbul_ISE$TL.BASED.ISE)
sum.istanbul_ISE_TL

# Summary of the S&P500 returns
sum.SP <- summary(istanbul_ISE$SP)
sum.SP

# Summary of the DAX returns
sum.DAX <- summary(istanbul_ISE$DAX)
sum.DAX

# Summary of the FTSE returns
sum.FTSE <- summary(istanbul_ISE$FTSE)
sum.FTSE

# Summary of the NIKKEI returns
sum.NIKKEI <- summary(istanbul_ISE$imkb_x.NIKKEI)
sum.NIKKEI

# Summary of the BOVESPA returns
sum.BOVESPA <- summary(istanbul_ISE$BOVESPA)
sum.BOVESPA

# Summary of the EU returns
sum.EU <- summary(istanbul_ISE$EU)
sum.EU

# Summary of the EM returns
sum.EM <- summary(istanbul_ISE$EM)
sum.EM

# Proportion of positive USD Based returns
propPositive <- sum(istanbul_ISE$USD.BASED.ISE > 0) / length(istanbul_ISE$USD.BASED.ISE)
propPositive

# Proportion of positive TL Based returns
propPositive2 <- sum(istanbul_ISE$TL.BASED.ISE > 0) / length(istanbul_ISE$TL.BASED.ISE)
propPositive2

# histogram of USD Based returns of ISE
hist(istanbul_ISE$USD.BASED.ISE,
        #xlim = c(-.125, .125),
        main="Histogram of USD Based Returns of ISE from 2009 to 2011",
        ylab="Count",
        xlab="Returns",
        breaks=100,
        col="green")$breaks

# histogram of ISE returns
hist(istanbul_ISE$TL.BASED.ISE,
     #xlim = c(-.125, .125),
     main="Histogram of TL Based Returns of ISE from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")$breaks

# histogram of S&P500 returns
hist(istanbul_ISE$SP,
     #xlim = c(-.125, .125),
     main="Histogram of S&P500 Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# histogram of DAX returns
hist(istanbul_ISE$DAX,
     #xlim = c(-.125, .125),
     main="Histogram of DAX Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# histogram of FTSE returns
hist(istanbul_ISE$FTSE,
     #xlim = c(-.125, .125),
     main="Histogram of FTSE Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# histogram of NIKKEI returns
hist(istanbul_ISE$imkb_x.NIKKEI,
     #xlim = c(-.125, .125),
     main="Histogram of NIKKEI Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# histogram of BOVESPA returns
hist(istanbul_ISE$BOVESPA,
     #xlim = c(-.125, .125),
     main="Histogram of BOVESPA Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# histogram of EU returns
hist(istanbul_ISE$EU,
     #xlim = c(-.125, .125),
     main="Histogram of EU Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# histogram of EM returns
hist(istanbul_ISE$EM,
     #xlim = c(-.125, .125),
     main="Histogram of EM Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

mmnorm.ISE_TL <- (istanbul_ISE$TL.BASED.ISE - min(istanbul_ISE$TL.BASED.ISE))/
                          (max(istanbul_ISE$TL.BASED.ISE) - min(istanbul_ISE$TL.BASED.ISE))
mmnorm.ISE_TL


sum.mmnorm.ISE_TL <- summary(mmnorm.ISE_TL)
sum.mmnorm.ISE_TL

zscore.ISE_TL <- (istanbul_ISE$TL.BASED.ISE - mean(istanbul_ISE$TL.BASED.ISE))/
                          sd(istanbul_ISE$TL.BASED.ISE)
zscore.ISE_TL

sum.Zscore.ISE_TL <- summary(zscore.ISE_TL)
sum.Zscore.ISE_TL

# histogram of ISE_TL Z-Score returns
hist(zscore.ISE_TL,
    # xlim = c(-.2, .2),
     main="Histogram of ISE TL Based Z-Score Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

istanbul_ISE_TL.skewness <- (3*(mean(istanbul_ISE$TL.BASED.ISE) - median(istanbul_ISE$TL.BASED.ISE)))/
                                  sd(istanbul_ISE$TL.BASED.ISE)
istanbul_ISE_TL.skewness

testDecSCaleInv <- istanbul_ISE$TL.BASED.ISE*100
testDecSCaleInv

sum.testDecSCaleInv <- summary(testDecSCaleInv)
sum.testDecSCaleInv

# histogram of ISE_TL Z-Score returns
hist(testDecSCaleInv,
     # xlim = c(-.2, .2),
     main="Histogram of ISE TL Based Percentage Returns from 2009 to 2011",
     ylab="Count",
     xlab="Returns",
     breaks=100,
     col="green")

# Binning
n <- length(istanbul_ISE$TL.BASED.ISE)
whichbin <- c(rep(0, n))
nbins <- 3


#k-means clustering
data <- istanbul_ISE$TL.BASED.ISE
kmeansclustering <- kmeans(data, centers=nbins)
whichbin <- kmeansclustering$cluster
whichbin

#binning by 3 arbitrary cuts and creating a frequency distribution
whichbincuts = cut(whichbin, 3)
whichbincuts.freq = table(whichbincuts)
whichbincuts.freq

#Equal widthbinning by 5 arbitrary cuts and creating a frequency distribution
tableData = cut(data, 5) #, labels = c("Low", "Med-Low", "Med", "Med-High", "High")
tableData.freq = table(tableData)
tableData.freq


#create frequency distrbution
newbincuts <- cut(range_TL_Based, 26)
newbinfreq = table(newbincuts)
newbinfreq

#std of ISE TL
dataStDev <- sd(istanbul_ISE$TL.BASED.ISE)
dataStDev

# attempt for normal distribution but contains negative numbers
natlog_ISE <- log(istanbul_ISE$TL.BASED.ISE)
natlog_ISE

sqrt_ISE <- sqrt(istanbul_ISE$TL.BASED.ISE)
sqrt_ISE

invsqrt_ISE <- 1/(sqrt(istanbul_ISE$TL.BASED.ISE))
invsqrt_ISE

# Regression Analysis between the ISE, and the S&P500 and NIKKEI
tlBasedReturns <- istanbul_ISE$TL.BASED.ISE
SP <- istanbul_ISE$SP
date <- as.Date(istanbul_ISE$date, format="%d/%m/%Y")

lm.out <- lm(tlBasedReturns~SP)

plot(SP, 
     tlBasedReturns, 
     #xlim = c(-.03, .03),
     main="ISE TL Returns in Relation to S&P500",
     xlab="S&P",
     ylab="ISE TL Returns")
abline(lm.out)
summary(lm.out)

nikkei <- istanbul_ISE$imkb_x.NIKKEI

mreg.out <- lm(tlBasedReturns ~ SP + nikkei)
summary(mreg.out)
lm.out2 <- lm(tlBasedReturns~nikkei)
plot(nikkei, 
     tlBasedReturns,
     main="ISE TL Returns in Relation to NIKKEI",
     xlab="",
     ylab="ISE TL Returns")

abline(lm.out2)
summary(lm.out2)
