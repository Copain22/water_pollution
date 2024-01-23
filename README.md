Introduction

Since we have high levels of pollution in nearly half of US (United States) rivers and more than one-third of lakes are polluted and unfit for swimming, fishing, and drinking. Water pollution is damaging to living species, and it causes a variety of illnesses. It has caused damage to the aquatic ecosystem, necessitating the search for potential solutions. Several global laws and guidelines have been established to protect water quality. 	This would also serve as a supporting tool for government and agencies on decision-making for water resources.  In Florida, “CERP dictates a complex funding arrangement whereby the federal and state governments shared construction costs equally, subjecting the plan to the unpredictable nature of the legislative budgeting process at both the federal and state levels” 
The goal for our project is to build a model for water quality prediction using a dataset of Cape Coral, Florida and give better insights on what range of chemical factors contribute on deciding the water quality levels and how much contaminated they are.
The selected dataset contains 18148 observations(records/entities) with 48 variables (attributes). Considering the more attributes, a dataset has lessened the chances that the predictive model would be stable. The initial filtering of attributes was performed by considering the standards defined by the World Health Organization (WHO). The following table shows the list of attributes selected which have a higher degree of importance in measuring water quality, and their range of values indicating a good quality of water.

In the following section are describe can be found an analysis of data to understand better their characteristics. Then we try to understand if there is a specific pattern for the missing values, which will then be helpful to define the model that should be used for the data imputation. After imputing the missing data, the next phase concerns balancing. Since one of the reasons for bias in machine learning models is unbalanced data, we try resolve this issue by trying several algorithms. The final step is building the machine learning models that predict whether the quality of water is “poor” or “good”.

Project Workflow

To implement this project, we followed specific steps. Initially we had to define the dataset to be used. The second step consisted in working with the dataset which consist of: selecting the important features, impute missing data and balance the water quality categories. For each of the steps of working with the dataset, several models were tested and the best performing one is chosen. Next, several training models are tested, such as naïve bayes, knn, random forest and finally artificial neural network. The project workflow is shown in figure 1.

 ![image](https://github.com/Copain22/water_pollution/assets/120750628/459ca19b-850f-4023-af9d-7d0f823866e7)

Figure 1: Project workflow

Dataset Description

We have explored three datasets for testing water quality, they are:
•	https://hub.arcgis.com/datasets/b0579ba7aa1145e090c3a74e295564df/explore?location=26.646897%2C-81.929073%2C11.71&showTable=true
•	https://data.doi.gov/dataset/water-quality-data
•	https://redivis.com/datasets/g8je-49bxaag3y
Among these websites we have selected data provided by the city of Cape Coral because it has more significant features and helps to predict our model more accurately. The data set contains 18148 observations with 48 variables, it also has missing values 229 in water temperature, 385 in specific conductance, 299 in dissolved oxygen, 567 in pH, 293 in sampling depth, 303 in salinity, 18148 in dissolved oxygen saturation, 6956 in total dissolved solids, 12073 in Fecal coliforms, 18148 in organic nitrogen,15858 in volatile suspended solids, 15893 in volatile dissolved solids. Among all the features we need to select more significant and eliminate remaining fields. The data set contains 18148 observations with 48 variables.
Data Pre-processing
Before delving into defining and building the models for predicting the water quality, it is important to initially understand the data and perform necessary data clean up and preparation. 
The selected dataset contains 18148 observations(records/entities) with 48 variables (attributes). Considering the more attributes, a dataset has lessened the chances that the predictive model would be stable. The initial filtering of attributes was performed by considering the standards defined by the World Health Organization (WHO). The following table shows the list of attributes selected which have a higher degree of importance in measuring water quality, and their range of values indicating a good quality of water. 

Attributes	Description	Threshold Values
Water Temperature	Temp of water collected at certain point of time	30-35 Celsius
Specific Conductance	Collective concentration of ions in water	<=50
Dissolve Oxygen	Amount of oxygen present in water	<6
Ph	Measure of acididty/basicity in water	6.5-8.0
NH3	Amount of combined nitrogen and hydrogen present	<0.2
NO2	Amount of Nitrites present in water	<1
NO3	Amount of Nitrates present in water	<10
Salinity	Salt content dissolved in water	<7
Total Kjeldahl Nitrogen	Total sum of ammonia nitrogen and organic nitrogenous compounds	<1
Total Nitrogen	Amount of Nitrites and Nitrates contaminant present in water	<10
Total Phosphates	Total amount of phosphorus present in water	<10
Bioloxd	Dissolved gas molecules held in water	<5
Alkalinity	Amount of neutralizing power of acids and bases	<200
Fecal Coliforms	Amount of harmful pathogens present in water	<200

Selecting Features on basis of domain knowledge: There are different chemical components present in the water with multiple combinations that affect the quality of water. So, by intruding further with the domain knowledge specific to chemicals we have taken 12 variables refined out of 48 as those features are considered important as per the bio chemist’s knowledge and reference articles [1] related to water quality and its contamination. Also, the lab recorded water samples by force fitting the chemicals are eliminated for model training as it might create a bias while training the models.

 ![image](https://github.com/Copain22/water_pollution/assets/120750628/3c82af24-0488-45a2-b110-66ebc6fb9ab5)

Figure 2: Attribute correlation heatmap

Defining a Target Variable  
For defining the target variable, we used the water quality standards defined by the WHO and created a new variable named water quality which has two attributes “good” and “bad”. The new variable was computed using:
-	Conduct < 50.0  
-	Dissolved Oxygen < 6  
-	6.5 <= PH  <= 8.0 
-	Salinity < 7 
-	NH3 < 0.2 
-	Total Kjeldahl Nitrogen < 1.0  
-	NO2 < 1 
-	NO3 < 10 
-	Total Nitrogen < 10 
-	Total Phosphate <10 
-	Bioloxd < 5 
-	Alakinity < 200 
-	Fecal coliform < 200

Handling Missing Values
As presented earlier in this report, in the selected dataset are detected missing variables in some of the attributes. Following, an information about the number of missing values for each attribute, taking in consideration that the overall dataset consists of 18, 148 observations.

![image](https://github.com/Copain22/water_pollution/assets/120750628/a74da37a-2add-4b64-b27b-e9609a44b669)

Figure 3: Missing data for each attribute

The following bar chart shows the completeness of each attribute. Here it can be identified that the attribute with the most missing data is FECCOLI, where more than 50% of values are missing.

                      


Next, we try to identify any correlation of missing values between variables. The following heat map serves this purpose. As it is shown, there are several pairs of attributes that seem to be correlated regarding the missing values, such as TOT_KJN and NH3_NITR, or NO2_NITR and NH3_NITR. We can say that the missing values are not completely at random, however we cannot identify a specific pattern. Therefore, we are assuming that we are dealing with MAR data.
                                Figure 4: Missing data heatmap

Denoising Autoencoder

The Denoising Autoencoder consists of multiple layers starting from encoder given with input dimensions and activation layer – ‘rectified linear unit’ along with input data. Decoder with the same number of input dimensions and activation layer as linear along with encoded input. The dataset is spitted in 80% training and 20% testing. The resulting loss is: 6.7029e-07 and the dataset has no missing data. These resulted in the 17164 observations that recorded poor water and 984 as good water.

KNN Impute

Next, we have implemented KNN Impute by defining the k neighbours as 4 and train the model. The K-Nearest Neighbours algorithm is used by the KNN Impute method to fill in missing numbers in a dataset. KNN Impute assumes that the missing values are missing at random (MAR), and the estimated values assume that the missing values are like the observed values. This rule set reported in 16708 observations as poor water and 1440 as good.

Soft Impute
Soft Impute is a method for filling in missing data in matrices. Soft Impute can deal with missing data in many ways, such as data that is missing totally at random (MCAR), missing at random (MAR), or missing not at random (MNAR). Soft Impute can also handle datasets with a lot of missing data and can be used to fill in missing values for both continuous and categorical data. The execution of this method gave a distribution of 15618 observations as poor water and 2350 as good water.
For our case, Denoising Auto Encoder worked better as it gave a RMSE (Root Mean Squared Error) value better than other imputation methods.
Method	RMSE
Denoising Auto Encoder 	2.2
Soft Impute	0.0
KNN Impute	0.0
In the following image is shown the distribution of values for each attribute selected, after the imputation.

![image](https://github.com/Copain22/water_pollution/assets/120750628/394758e8-2a4d-4c5e-97b2-164a90de4bd2)

  
Figure 5: Value distribution for each attribute 
Scaling Features 
Minmax Scalar () function: Defining Train and Test Set: The dataset is split into training and testing dataset with 80% of train data and 20% of test data.

Handling Imbalanced Data
SMOTE 
It stands for Synthetic Minority Over-sampling Technique. SMOTE works by creating synthetic examples of the minority class by interpolating between existing minority class samples.
Undersampling
In undersampling, the majority class is reduced by randomly removing some of its instances from the training dataset, so that the number of instances in the majority class is similar to the number of instances in the minority class. We investigated different techniques to deal with imbalanced data, and in our situation, we found that SMOTE was causing overfitting, while undersampling produced better results.

Predicting Models
We tested seven predictive models and found the artificial neural network to have acceptable parameters. Details and results for each model are provided below.
Bagging Classifier
This approach used in machine learning to address imbalanced datasets where one class has considerably fewer samples than the other. It is a data augmentation technique that generates synthetic examples of the minority class by producing new samples that fall between existing minority class samples. This technique aims to create a more balanced class distribution by providing additional representative samples for the minority class. Ultimately, using SMOTE can enhance machine learning models' performance when working with imbalanced datasets.


Ensemble Method (Bagging Classifier + SVM)
We utilized the f_classif score to choose the most important features and assigned weights to two classes. Then, we developed an SVM model to tackle overfitting and enhance the model's stability. Additionally, we created a bagging classifier model and conducted under sampling to balance the class distribution. Lastly, we trained the model.
The Accuracy of Ensemble turned out to be 86.2% which seems to be decent prediction as compared to the above SMOTE method and could be improved by tuning of hyperparameters with different range of values.
However, we will be working on this data with certain models that gives best results for the imbalanced dataset. 

Naïve Bayes

Naive Bayes is a probabilistic algorithm used in machine learning for classification. It is based on Bayes' theorem and the idea that traits don't depend on each other. It's called "naive" because it assumes that each trait is separate from the others. The idea that features are independent can sometimes be taken too far, which makes the program less accurate than others.

KNN

KNN, which stands for "K-Nearest Neighbours," is a method for machine learning that can be used for both classification and regression. It is a non-parametric method, which means that it doesn't make any assumptions about how the data are spread out. 

Random Forest

Random forest is a method for machine learning that is used for both classification and regression tasks when it is supervised. It is a type of learning that uses various decision trees to improve the model's ability to predict. The algorithm works by making a forest of decision trees. We have got an accuracy of 99.9% which is an overfit model in predicting the water quality.
Artificial Neural Network
The ANN model architecture, suggested by Rustam et al. (2022), uses an initial activation layer followed by a dropout layer and another activation layer with a dropout. The model employs a ReLU activation layer after the dense layer of 256 neurons and a 0.5 dropout rate to randomly eliminate 50% of neurons. The second dense layer has 256 neurons, followed by a ReLU activation layer and a 0.5 dropout layer. The model predicts water quality using a dense two-neuron layer, Adam optimizer, and binary crossentropy loss function, achieving 95% accuracy after being trained for 20 epochs.

 
