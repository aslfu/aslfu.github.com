---
layout: post
title: "The Analytics Edge课程小结"
date: 2015-05-31 23:00:32 +0800
comments: true
categories: 
---
##  Intro
~~~r
str()
summary()
tapply(arg1,arg2,arg3) #group arg1 by arg2, apply arg3
~~~
<!--more-->
## Linear Regression

**多变量线性模型（x,y均为连续变量)**
   
$y^i = \beta_0 + \beta_1X_1^i + \beta_2X_2^i+ ... + \beta_kX_k^i + \epsilon^i$

~~~r
fit <- lm(Price ~ AGST + HarvestRain, data=wine) 

>fit$coefficients
 (Intercept)         AGST  HarvestRain 
-2.202653601  0.602616906 -0.004570055 
    
>fit$residuals # response-fitted values 理想情况下，残差是正态分布
>fit$fitted.values # fitted mean values
~~~
~~~r
#作图查看模型效果
plot(fit)
#pic1 Residuals vs Fitted 
#检查拟合度
#pic2 Normal Q-Q
#检查残差是否为正态分布，理想情况下，残差符合标准正态分布。若存在明显趋势，则可能是模型遗漏了信息
#pic3 Scale-Location
#标准化残差开方散点图，显示异常点
~~~
~~~r
>summary(fit)
~~~
- **Coefficients**

**Estimate**
由最小二乘法计算出的参数估计值 $ \hat{\beta_k}$ 。参数为0，代表该变量对模型没有贡献。考虑参数真的为0时（Null Hypothesis），出现该估计值的概率有多大，即显著性检验。  
**t value** $ t = Estimate/Std. Error$ 越大越显著  
**Pr(>|t|)** 越小越显著  
**Std.Error** ＝ Standard Error 与样本/总体标准差成正比，样本大小成反比

- **R-squared**  

衡量模型拟合质量的指标，一般来说越大越好。Compares the best model to a "baseline" model(does not use any variables)  

$ R-square= \frac{SSR}{SST} = 1- \frac{SSE}{SST}$

SSR 预测数据与原始数据均值之差的平方和  
SSE 预测数据与原始数据对应点的误差的平方和  
SST 原始数据和原始数据均值之差的平方和

*Adjusted R-squared*  调整的R平方考虑了变量数目。增加变量，R平方一定增加，但是模型复杂容易过拟合。

- 多重共线性  
>Highly correlated independent variables can affect the interpretation of the coefficients.We won't worry about dealing with correlated independent variables here, but if interpreting the coefficients is important, multicollinearity should be addressed.


- 模型选择  
$ AIC = n + n\log{2\pi} + n\log{(RSS/n)} + 2(p+1)$  
$ BIC = n + n\log{2\pi} + n\log{(RSS/n)} + (\log{n})(p+1)$  
......

## Logistic Regression
####一、缺失值处理
- mice包 多重插补法  

>多重插补(Multiple Imputation)是用于填补复杂数据缺失值的一种方法，该方法通过变量间关系来预测缺失数据，利用蒙特卡罗随机模拟方法生成多个完整数据集，再对这些数据集分别进行分析，最后对这些分析结果进行汇总处理。FSC是基于链式方程的插补方法，因此也称为MICE (Multiple Imputation by Chained Equations )。它与其他多重插补算法的本质区别是，它在进行插补时不必考虑被插补变量和协变量的联合分布，而是利用单个变量的条件分布逐一进行插补。在R语言中通过程序包mice中的函数mice()可以实现该方法，它随机模拟多个完整数据集并存入imp，再对imp进行线性回归，最后用pool函数对回归结果进行汇总。
>


~~~r
# Install and load mice package
install.packages("mice")
library(mice)

# Multiple imputation
set.seed(144)
imputed = complete(mice(dataset))

~~~
####二、划分训练集和测试集

- 分类问题，平衡训练集与测试集中因变量的比例

~~~r
library(caTools)
# Randomly split the data into training and testing sets
set.seed(1000)
split = sample.split(quality$PoorCare, SplitRatio = 0.65)

# Split up the data using subset
train = subset(quality, split==TRUE)
test = subset(quality, split==FALSE)
~~~
- 非分类问题  

~~~r
set.seed(1000)
#sample(x,size,replace = FALSE, prob=NULL)
split = sample(nrow(quality),0.7*nrow(quality))
train = qualitysplit,]
test = quality[-split,]
~~~
####三、广义线性模型
~~~r
glm(formula, family = familytype,data)
#Family  Default Link Function
#binomial(link = "logit")
#gaussian(link = "identity")
#Gamma(link = "inverse")
#inverse.gaussian(link = "1/mu^2")
#poisson(link = "log")
#quasi(link = "identity", variance = "constant")
#quasibinomial(link = "logit")
#quasipoisson(link = "log")
~~~
- 逻辑回归  

~~~r 
#Logistic Regression Model
QualityLog = glm(PoorCare ~ OfficeVisits + Narcotics, data=train, family=binomial)

# Predictions on the test set
predictTest = predict(QualityLog, type="response", newdata=test)
predictTrain = predict(QualityLog, type="response", newdata=train)
~~~
####  四、二元分类模型阈值问题

- **Receiver Operator Characteristic（ROC曲线)** 
- 横轴为False positive rate（在所有实际为0的样本中，被错误地判断为1的比率)  
$ TPR = TP/(TP+FN)$  
$ Sensitivity=\frac{TP}{TP+FN} $  
- 纵轴为True positive rate（在所有实际为1的样本中，被正确地判断为1的比率）  
$FPR = FP/(FP+TN)$  
$Specificity = \frac{TN}{TN+FP}$   

>将sensitivity与specificity结合，改变阈值，获得多对tpf和fpf值，绘制ROC曲线，寻找最佳阈值

~~~r
# Install and load ROCR package
install.packages("ROCR")
library(ROCR)

# Prediction function
# prediction(predictions,labels)
ROCRpred = prediction(predictTrain, train$PoorCare)

# Performance function
#performance(prediction.obj,measure)
ROCRperf = performance(ROCRpred, "tpr", "fpr")

#get auc
as.numeric(performance(ROCRpred, "auc")@y.values)

# Plot ROC curve
plot(ROCRperf)

# Add colors
plot(ROCRperf, colorize=TRUE)

# Add threshold labels 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
~~~
- AUC值的计算  
>AUC（Area Under Curve）被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

- AUC意味着什么  
>
那么AUC值的含义是什么呢？根据(Fawcett, 2006)，AUC的值的含义是：The AUC value is equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.
>
首先AUC值是一个概率值，当你随机挑选一个正样本以及一个负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值。当然，AUC值越大，当前的分类算法越有可能将正样本排在负样本前面，即能够更好的分类。  
>
>ROC曲线有个很好的特性：当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。
 
## Trees
~~~r
library(rpart)
library(rpart.plot)
#rpart(formula, data=, method=, control=)
# method = "class" for a classification tree
#          "anova" for a regression tree
# parms = For classification splitting, the list can contain any #of: the vector of prior probabilities (component prior), the loss matrix (component loss) or the splitting index (component split). The priors must be positive and sum to 1. The loss matrix must have zeros on the diagonal and positive off-diagonal elements. 
# control = optional parameters for controlling tree growth. For example, control=rpart.control(minsplit=30, cp=0.001) requires that the minimum number of observations in a node be 30 before attempting a split and that a split must decrease the overall lack of fit by a factor of 0.001 (cost complexity factor) before being attempted.
~~~
#### 一、Grow a tree
- CP(complexity parameter,值越小，模型越复杂)   
$ minimize \sum(RSS \space at \space each \space leaf) +  \lambda S$  
$cp = \frac{\lambda}{RSS(no \space splits)} $

~~~r
# Load CART packages
library(rpart)
library(rpart.plot)

# CART model
latlontree = rpart(MEDV ~ LAT + LON, data=boston)
prp(latlontree)
~~~

~~~r
# Simplify tree by increasing minbucket
latlontree = rpart(MEDV ~ LAT + LON, data=boston, minbucket=50)
~~~
#### 二、Cross Validation
~~~r
# Load libraries for cross-validation
library(caret)
library(e1071)

# Number of folds
tr.control = trainControl(method = "cv", number = 10)

# cp values
cp.grid = expand.grid( .cp = (0:10)*0.001)

# Cross-validation
tr = train(MEDV ~ LAT + LON + CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO, data = train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)

# Extract tree
best.tree = tr$finalModel
prp(best.tree)

# Make predictions
best.tree.pred = predict(best.tree, newdata=test)
best.tree.sse = sum((best.tree.pred - test$MEDV)^2)
~~~
#### 三、Random Forest
~~~r
# Install randomForest package
install.packages("randomForest")
library(randomForest)

# Convert outcome to factor
Train$Reverse = as.factor(Train$Reverse)
Test$Reverse = as.factor(Test$Reverse)

# Build random forest model
StevensForest = randomForest(Reverse ~ Circuit + Issue + Petitioner + Respondent + LowerCourt + Unconst, data = Train, ntree=200, nodesize=25 )

# Make predictions
PredictForest = predict(StevensForest, newdata = Test)
~~~

## Text Analytics

- stringsAsFactors  
data.frame默认stringsAsFactors＝TRUE

~~~r
# Read in the data

tweets = read.csv("tweets.csv", stringsAsFactors=FALSE)

# Create dependent variable

tweets$Negative = as.factor(tweets$Avg <= -1)
~~~
- tm：文本挖掘库  

~~~r
# Install new packages

install.packages("tm")
library(tm)
~~~
- 转换成语料库Corpus, 指一系列文档的集合，是tm包管理文件的数据结构

> 在tm包中，Corpus可以分为两种。一种是Volatile Corpus，这种数据结构是作为R对象保存在内存中,使用VCorpus()或者Corpus()函数；另一种就是Permanent Corpus，作为R的外部保存，使用PCorpus()函数。显然，如何选择取决于内存大小以及运算速率的要求了。
> 

~~~r

# Create corpus
 
corpus = Corpus(VectorSource(tweets$Tweet))
~~~
- 查看Corpus  

~~~r
corpus #显示文档数据数量
names(corpus) #文档名称
inspect(corpus[1]) #提取第一篇文档的完整信息
corpus[[1]] #提取第一个文档
~~~ 
- 信息转化  
tm_map()函数可以将转化函数作用到每一个文档数据上

~~~r

# Convert to lower-case
corpus = tm_map(corpus, tolower)
#转化为纯文本
corpus = tm_map(corpus, PlainTextDocument)
# Remove punctuation
corpus = tm_map(corpus, removePunctuation)
#去除特殊字符
for (i in seq(corpus)) {
    docs[[i]] <- gsub("/", " ", corpus[[i]])
    docs[[i]] <- gsub("@", " ", corpus[[i]])
    docs[[i]] <- gsub("-", " ", corpus[[i]])
}
#去除数字
corpus <- tm_map(corpus,removeNumbers)
#去除多余的空格
corpus <- tm_map(corpus,stripWhitespace)
# Look at stop words 
stopwords("english")[1:10]
# Remove stopwords and apple
corpus = tm_map(corpus, removeWords, c("apple", stopwords("english")))

# Stem document  词干化
install.packages("SnowballC")
library(SnowballC)
corpus = tm_map(corpus, stemDocument)
~~~
- 创建词条－文档关系矩阵  
行是词条，列是文档，矩阵中每个值表示对应的词条在对应的文档中出现的次数

~~~r
# Create matrix

frequencies = DocumentTermMatrix(corpus)

# Look at matrix 

inspect(frequencies[1000:1005,505:515])
~~~
- 删减稀疏条目

~~~r

# Check for sparsity

findFreqTerms(frequencies, lowfreq=20)

# Remove sparse terms 删除稀疏程度高于0.995的条目

sparse = removeSparseTerms(frequencies, 0.995)
~~~
- 分类模型  

~~~r
# Convert to a data frame

tweetsSparse = as.data.frame(as.matrix(sparse))

# Make all variable names R-friendly

colnames(tweetsSparse) = make.names(colnames(tweetsSparse))

# Add dependent variable

tweetsSparse$Negative = tweets$Negative

# Split the data

library(caTools)

set.seed(123)

split = sample.split(tweetsSparse$Negative, SplitRatio = 0.7)

trainSparse = subset(tweetsSparse, split==TRUE)
testSparse = subset(tweetsSparse, split==FALSE)


~~~
- 决策树

~~~r
# Build a CART model

library(rpart)
library(rpart.plot)

tweetCART = rpart(Negative ~ ., data=trainSparse, method="class")

# Evaluate the performance of the model
predictCART = predict(tweetCART, newdata=testSparse, type="class")
~~~
- 随机森林

~~~r
# Random forest model

library(randomForest)
set.seed(123)

tweetRF = randomForest(Negative ~ ., data=trainSparse)

# Make predictions:
predictRF = predict(tweetRF, newdata=testSparse)
~~~

## Clustering
 
主要方法；层次聚类（Hierarchical clustering）与k均值聚类（K-Means clustering）  
### **一、层次聚类**  
层次聚类首先将每个样本单独作为一类，然后将不同类之间距离最近的进行合并，合并后重新计算类间距离。  

>可用于定义“距离”的统计量包括了欧氏距离(euclidean)、马氏距离(Mahalanobis)、 曼哈顿距离(Manhattan)、两项距离(binary)、明氏距离(minkowski)。还包括相关系数和夹角余弦。
  
>在计算类间距离时则有六种不同的方法，分别是最短距离法、最长距离法、类平均法、重心法、中间距离法、离差平方和法。

~~~r
movies = read.table("movieLens.txt", header=FALSE, sep="|",quote="\"")

# Compute distances
distances = dist(movies[2:20], method = "euclidean")

# Hierarchical clustering
clusterMovies = hclust(distances, method = "ward") 

# Plot the dendrogram
plot(clusterMovies)

# Assign points to clusters 10类
clusterGroups = cutree(clusterMovies, k = 10)
~~~
### **二、k均值聚类**  
>K均值聚类又称为动态聚类，它的计算方法较为简单，也不需要输入距离矩阵。首先要指定聚类的分类个数N，随机取N个样本作为初始类的中心，计算各样本与类中心的距离并进行归类，所有样本划分完成后重新计算类中心，重复这个过程直到类中心不再变化。

~~~r
healthy = read.csv("healthy.csv", header=FALSE)
healthyMatrix = as.matrix(healthy)

# Plot image
image(healthyMatrix,axes=FALSE,col=grey(seq(0,1,length=256)))

# Specify number of clusters
k = 5

# Run k-means
set.seed(1)
KMC = kmeans(healthyVector, centers = k, iter.max = 1000)
str(KMC)

# Extract clusters
healthyClusters = KMC$cluster
KMC$centers[2]

# Plot the image with the clusters
dim(healthyClusters) = c(nrow(healthyMatrix), ncol(healthyMatrix))

image(healthyClusters, axes = FALSE, col=rainbow(k))
~~~

~~~r
# Apply to a test image
 
tumor = read.csv("tumor.csv", header=FALSE)
tumorMatrix = as.matrix(tumor)
tumorVector = as.vector(tumorMatrix)

# Apply clusters from before to new image, using the flexclust package
install.packages("flexclust")
library(flexclust)

KMC.kcca = as.kcca(KMC, healthyVector)
tumorClusters = predict(KMC.kcca, newdata = tumorVector)

# Visualize the clusters
dim(tumorClusters) = c(nrow(tumorMatrix), ncol(tumorMatrix))

image(tumorClusters, axes = FALSE, col=rainbow(k))
~~~
## Visualization
>ggplot2 将绘图视为一种映射，即从数学空间映射到图形元素空间。例如将不同的数值映射到不同的色彩或透明度。该绘图包的特点在于并不去定义具体的图形（如直方图，散点图），而是定义各种底层组件（如线条、方块）来合成复杂的图形，这使它能以非常简洁的函数构建各类图形，而且默认条件下的绘图品质就能达到出版要求。  

####ggplot2基本要素####


- **数据（Data）和映射（Mapping)**  
	
	ggplot(data,mapping) 把数据映射到图形属性

~~~r
	install.packages("ggplot2")
	library(ggplot2)

	# Create the ggplot object with the data and the aesthetic mapping:
	p<-ggplot(WHO, aes(x = GNI, y = FertilityRate, color = Region)) 
~~~

- **几何对象（Geometric）**

~~~r
# Add the geom_point geometry
p + geom_point()

# Make a line graph instead:
p + geom_line()

# Redo the plot with blue triangles instead of circles:
p + geom_point(color = "blue", size = 3, shape = 17) 

#geom_abline 		geom_area 	
#geom_bar 			geom_bin2d
#geom_blank 		geom_boxplot 	
#geom_contour 		geom_crossbar
#geom_density 		geom_density2d 	
#geom_dotplot 		geom_errorbar
#geom_errorbarh 	geom_freqpoly 	
#geom_hex 			geom_histogram
#geom_hline 		geom_jitter 	
#geom_line 			geom_linerange
#geom_map 			geom_path 	
#geom_point 		geom_pointrange
#geom_polygon 		geom_quantile 	
#geom_raster 		geom_rect
#geom_ribbon 		geom_rug 	
#geom_segment 		geom_smooth
#geom_step 			geom_text 	
#geom_tile 			geom_violin
#geom_vline
~~~

- **标尺（Scale）**  

	在对图形属性进行映射之后，使用标尺可以控制这些属性的显示方式，比如坐标刻度，可能通过标尺，将坐标进行对数变换；比如颜色属性，也可以通过标尺，进行改变。

~~~r
# Let's try a log transformation:
ggplot(WHO, aes(x = log(FertilityRate), y = Under15,color = Region)) + geom_point()
# or
ggplot(WHO, aes(x = FertilityRate, y = Under15,color = Region)) + geom_point()+scale_x_log10()+scale_color_manual(values=rainbow(7))
~~~

- **统计变换（Statistics）**
对原始数据进行某种计算，然后在图上表示出来，ggplot2提供的统计变幻方式有：  

~~~r
stat_abline       stat_contour      stat_identity     stat_summary
stat_bin          stat_density      stat_qq           stat_summary2d
stat_bin2d        stat_density2d    stat_quantile     stat_summary_hex
stat_bindot       stat_ecdf         stat_smooth       stat_unique
stat_binhex       stat_function     stat_spoke        stat_vline
stat_boxplot      stat_hline        stat_sum          stat_ydensity
~~~

~~~r
# Simple linear regression model to predict the percentage of the population under 15, using the log of the fertility rate:
mod = lm(Under15 ~ log(FertilityRate), data = WHO)
summary(mod)

# Add this regression line to our plot:
ggplot(WHO, aes(x = log(FertilityRate), y = Under15)) + geom_point() + stat_smooth(method = "lm")

# 99% confidence interval
ggplot(WHO, aes(x = log(FertilityRate), y = Under15)) + geom_point() + stat_smooth(method = "lm", level = 0.99)

# No confidence interval in the plot
ggplot(WHO, aes(x = log(FertilityRate), y = Under15)) + geom_point() + stat_smooth(method = "lm", se = FALSE)

# Change the color of the regression line:
ggplot(WHO, aes(x = log(FertilityRate), y = Under15)) + geom_point() + stat_smooth(method = "lm", colour = "orange")

~~~

- **坐标系统（Coordinante）**

>坐标系统控制坐标轴，可以进行变换，例如XY轴翻转，笛卡尔坐标和极坐标转换，以满足我们的各种需求。  
>
>坐标轴翻转coord_flip()  
>转换成极坐标coord_polar()

- **图层（Layer）**  
采用“＋”来添加图层，即可以是添加映射关系，也可以是多个数据集绘图
- **分面（Facet）**  
facet_wrap()
- **主题（Theme）**

~~~r
# Add a title to the plot:
scatterplot + geom_point(colour = "blue", size = 3, shape = 17) + ggtitle("Fertility Rate vs. Gross National Income")

# Save our plot:
fertilityGNIplot = scatterplot + geom_point(colour = "blue", size = 3, shape = 17) + ggtitle("Fertility Rate vs. Gross National Income")

pdf("MyPlot.pdf")

print(fertilityGNIplot)

dev.off()
~~~
####ggmap####
>ggmap包是基于ggplot2的图层语法构建的R包，它结合了来自Google Maps,OpenStreet Map,Stamen Maps和CloudMade Maps的静态地图信息来绘制主题地图。而且，ggmap中提供了一些应用函数以供使用者访问Google Geocoding,Distance Matrix和Directions 这几个 API。
>
> ggmap画地图的基本思路是下载地图作为ggplot2的基础画布，然后在其上加上数据，统计或者模型的分层。

~~~r
# Load the ggmap package
library(ggmap)

# Load in the international student data
intlall = read.csv("intlall.csv",stringsAsFactors=FALSE)

# Those NAs are really 0s, and we can replace them easily
intlall[is.na(intlall)] = 0

# Load the world map
world_map = map_data("world")
str(world_map)

# Lets merge intlall into world_map using the merge command
world_map = merge(world_map, intlall, by.x ="region", by.y = "Citizenship")
str(world_map)

# Plot the map
ggplot(world_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(fill="white", color="black") +
  coord_map("mercator")

# Reorder the data
world_map = world_map[order(world_map$group, world_map$order),]

# Redo the plot
ggplot(world_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(fill="white", color="black") +
  coord_map("mercator")

# Lets look for China
table(intlall$Citizenship) 

# Lets "fix" that in the intlall dataset
intlall$Citizenship[intlall$Citizenship=="China (People's Republic Of)"] = "China"

# We'll repeat our merge and order from before
world_map = merge(map_data("world"), intlall, 
                  by.x ="region",
                  by.y = "Citizenship")
world_map = world_map[order(world_map$group, world_map$order),]

ggplot(world_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("mercator")


# We can try other projections - this one is visually interesting
ggplot(world_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("ortho", orientation=c(20, 30, 0))

ggplot(world_map, aes(x=long, y=lat, group=group)) +
  geom_polygon(aes(fill=Total), color="black") +
  coord_map("ortho", orientation=c(-37, 175, 0))
~~~
