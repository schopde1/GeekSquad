# Predicting-Buildings-Energy-Efficiency

## Project Description 
### Problem Statement-
Ever growing population and progressive municipal business demands for constructing new buildings are known as the foremost contributor to greenhouse gases. Emission of greenhouse gases including carbon dioxide (CO2) in higher layers of the atmosphere are known as the main cause of global warming phenomena. Our project aims at targeting this problem and providing a solution to address this issue.

### Goal
Whether you are trying to reduce the cost of your energy bill or you're trying to reduce your carbon footprint, improving the energy efficacy of your building can both save you some money and even help the environment.The goal of this project is to create multiple regression models that come up with predictions for both the heating and cooling load of the building. We will take the best performing model and make suggestions on what factors to take into consideration while designing buildings with best energy efficiency.

### Dataset at a Glance

##### Source : [Dataset](https://www.kaggle.com/jarredpriester/predicting-a-building-s-energy-efficiency/data) (Tsanas, n.d.)

##### Data collection-
We perform energy analysis using 12 different building shapes . The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters 

##### Attribute Information:
The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
X1 Relative Compactness,X2 Surface Area,X3 Wall Area,X4 Roof Area,X5 Overall Height,X6 Orientation,X7 Glazing Area,X8 Glazing Area Distribution,y1 Heating Load,y2 Cooling Load

##### Shape : 768 samples and 8 features.

##### Dependent variable
Predict two real valued responses Heating and Cooling load.
These are thermal loads built onto a building’s HVAC system to maintain temperature in the building.


### Data Exploration and Preprocessing

##### Importing and Cleaning dataset
* Exploratory analysis to view Dataset- Reviewed Summary,Class,Str,Names of the dataset.
* Replacing column names with Feature names mentioned in variable description txt file.
* Checking NAN and Missing values using is.na function- Dataset did not have any missing value.


##### Scaling  data
In the machine learning algorithms if the values of the features are closer to each other there are chances for the algorithm to get trained well and faster but if they have high value difference it can take more time to understand the data and the accuracy will be lower. So, if the data in any conditions has data points far from each other, scaling is a technique to make them closer to each other or in simpler words, we can say that scaling is used for making data points generalized so that the distance between them will be lower. In our dataset we used the below steps to normalize the values-

* Using Scale function to Scale data with wide range of variation.
* Boxplot visualization for scaled and Unscaled data.
* Validate the data with Mean and SD to check if data is scaled.


##### Data visualization- Using GGPLOT function


* Scatter plot to visualize few predictors vs outcome.


##### Splitting the data set
* Used Train-test split- 70% Training,30% Test.



### VISUALIZATIONS CREATED DURING DATA PREPARATION
Used multiple data visualization library to Visualize relation of each component before and after scaling. Also used ggplot to create scatter plots to show relationship between few predictors and the dependent variables.


### MODELS IMPLEMENTED ON THE DATASET
The ML algorithms we used were Linear regression, random forest, K-nearest neighbors. We have also used deep learning models like Neural networks. Finally, we have used the ensemble model to bring together all the models implemented. 
We have used the RMSE value to detect efficiency. Models with least RMSE is the most suitable model.
#### Linear Regression
We use linear regression to estimate the dependence of dependent variable, such as Heating and cooling load, on all the predictors. Establishing a linear relationship between variables allows us to predict future energy use, and therefore estimate future potential energy savings from a project with some degree of accuracy.
##### Linear regression (RMSE) for Heating load- 1.88
##### Linear regression (RMSE) for Cooling load- 2.05

##### Visualizations-
Coefficients VS Predictors
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/LRcvsP.png)

Residuals-
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/LRResi.png)



#### K-Nearest Neighbour
The K-Nearest Neighbour or KNN has been widely used in classification and regression prediction problems owing to its simple implementation and outstanding performance. We feel that for our dataset this model can help derive the most efficient heating and cooling loads of the building.
##### KNN (RMSE) for Heating load- 1.33
##### KNN (RMSE) for Cooling load- 1.73


##### Visualizations-
K neighbors vs RMSE Cross validation(Heating)
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/KNHeating.png)
K neighbors vs RMSE Cross validation(Cooling)
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/KNcooling.png)


#### Random Forest
RF is an ensemble supervised machine learning algorithm. With multiple decision trees, each tree draws a sample random data giving the random forest more randomness to produce much better accuracy than decision trees.
##### RF (RMSE) for Heating load- 0.837
##### RF (RMSE) for Cooling load- 2.63

##### Visualizations-
Variable Importance Heating Load
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/RFHeating.png)
Variable Importance Cooling Load
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/RFCooling.png)



#### Neural Network
A simple neural network includes an input layer, an output (or target) layer and, in between, a hidden layer. The layers are connected via nodes, and these connections form a “network” – the neural network – of interconnected nodes. A node is patterned after a neuron in a human brain.
##### Heating load-
RMSE-2.621, Accuracy- 0.9834736
##### Cooling Load-
RMSE- 3.561743,Accuracy- 0.8814314
##### Visualizations
Neural net plot for Heating Load-
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/NNHeating.png)
Predicted vs Actual Heating Load-
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/PvsANNHeating.png)


Neural net plot for Cooling Load-
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/NNCooling.png)
Predicted vs Actual Cooling Load-
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/PvsANNCooling.png)


Heating Load Variable Importance-
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/HLVI.png)
Cooling Load Variable Importance
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/CLVI.png)



#### Ensemble Modelling-
Ensemble machine learning models can be proposed as an effective method for forecasting energy consumption in buildings. They combine the decisions from multiple models to improve the overall performance. Here we have used the Averaging technique as it is a regression problem.
RMSE for Heating Load-1.19, RMSE for Cooling Load-1.65

### RESULTS AND EVALUATION
On comparing the models implemented above we have plotted a bar graph with the RMSE values, and this shows us that the Ensemble model has the least RMSE for both Heating and colling load and therefore is the best fit for our dataset.
![](https://github.com/schopde1/Predicting-Buildings-Energy-Efficiency/blob/main/Images/Result.png)

### Business Insights
Machine learning analytics are gaining ground in the commercial buildings industry due to its cutting-edge ability to uncover patterns, produce accurate predictions, and automatically respond to those predictions. According to the U.S. Department of Energy, “as much as 30% of building energy consumption can be eliminated through more effective use of existing controls, and deployment of advanced controls,” which can be greatly enhanced by using machine learning.
As per our model we can suggest that the Ensemble model can be used on the building’s historical energy consumption data to reveal trends and predict future energy use. The longer an ML-driven analytics system is gathering data from a set of equipment, the more historical data it has on how, when, and why that equipment operates, allowing it to make more accurate predictions and better adjustments.

### Summary

Working on this project gave us a good hands-on and helped us understand and implement Predictive Analysis using Machine Learning Models.
Being able to perform Analysis by using different Machine Learning models, we were able to compare the models and choose the best one that was a good fit for our dataset.
