# Neural_Network_Charity_Analysis

Neural Networks and Deep Learning Models , Machine Learning classification, TensorFlow, Deep Neural Networks.

### Background:

Beck,a data scientist and programmer for a non-profit foundation "Alphabet Soup" asked us to help her foundation predict where to make investments.
Alphabate Soup are philonthropic foundation dedicated to helping organisations that protect the environment , improve people's well being and unify the world.Alphabate Soup has raised and dobated over $10B int he past 20 years. This money has been used to invest in lifesaving technology and organise reforestation groups around the world. Beck is in charge of data collection and analysis for the entire organisation. Her job is to analyse the impact of each donation and vet potential recipients. this helps to ensure that the foundations money is being used effectively.
Unfortunately, not every donation that comapny makes is impactful. In some cases , an organisation will take the money and disappear. As a result, Alphabet Soup's president has asked Becks to predict which organisations are worth donating to and which are too high risk? He wants her to create a mathematical data driven solution that can do this accurately.
Beck has decided that this problem is too complex for statistical and machine learning modelsthat she has used before.Instead she will design and train a deep learning neural network.This model will evaluate all types of input data and produce a clear decision making result.So we are helping Becks to learn neural networks and how to design and train these models using python tensorflow library. We will use our past experience with statistics and machine learning to help test and optimise our models. We will create a robust deep learning neural network capable of interpreting a large comlplex datasets and help Alphabet Soup to determine which organisation should receive their danations.

### Overview of the analysis:

- Compare the differences between the traditional machine learning classification and regression models and the neural network models.
- Describe the perceptron model and its components.
- Implement neural network models using TensorFlow.
- Explain how different neural network structures change algorithm performance.
- Preprocess and construct datasets for neural network models.
- Compare the differences between neural network models and deep neural networks.
- Implement deep neural network models using TensorFlow.
- Save trained TensorFlow models for later use.

### The purpose of this analysis:

A foundation, Alphabet Soup, wants to predict where to make investments. The goal is to use machine learning and neural networks to apply features on a provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The initial file has 34,000 organizations and a number of columns that capture metadata about each organization from past successful fundings.

### Results:

#### Data Preprocessing:

1.What variable(s) are considered the target(s) for your model?

Checking to see if the target is marked as successful in the DataFrame, indicating that it has been successfully funded by AlphabetSoup.Hence the IS_SUCCESSFUL column is considered the target for our model.

2.What variable(s) are considered to be the features for your model?

The APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, and ASK_AMT columns are considered to be the features of our model.

3.What variable(s) are neither targets nor features, and should be removed from the input data?

The EIN, NAME, STATUS, and SPECIAL_CONSIDERATION columns are neither targets nor features and should be removed from the input data.These columns will not increase the accuracy of the model and can be removed to improve code efficiency.

### Compiling, Training, and Evaluating the Model:

4.How many neurons, layers, and activation functions did you select for your neural network model, and why?

- I selected 120 neurons with a sigmoid function for my first layer, 100 nuerons with a ReLU function for the second, then 80 neurons with a tanh classification for the third and 60 for the fourth, and a sigmoid function for the outer layer. I chose to change the activation function for the first layer because it increased the model's performance.

![Alt_text](https://github.com/RGK73/Neural_Network_Charity_Analysis/blob/main/Images/optimisation.png)

![alt_text](https://github.com/RGK73/Neural_Network_Charity_Analysis/blob/main/Images/model_nn.png)

5.Were you able to achieve the target model performance?

The target was 75% and we could achieve 74.16%.I only achieved an accuracy of 74% and was not able to achieve the target model performance.

![Alt_text](https://github.com/RGK73/Neural_Network_Charity_Analysis/blob/main/Images/efficiency.png)

6.What steps did you take to try and increase model performance?

We tried to increase the model performance by dropping more columns, creating more bins for rare occurances in columns, decreasing the number of values in some bins, adding more neurons to the hidden layers, using a differnet activation function, and increasing the number of epochs.Desiging the model and creating a callback that saves the model's weights every 5 epochs.
We tried more hidden layers and increased epochs to 100 to increase model performance.Columns were reviewed and the STATUS and SPECIAL_CONSIDERATIONS columns were dropped as well as increasing the number of neurons and layers. Other activations were tried such as tanh.Linear activation produced the worst accuracy, around 28%. The relu activation at the early layers , tanh activation in the middle and sigmoid activation at the latter layers gave the best results.

### Summary: 

Recommendation:

A random forest model could solve this classification problem by randomly sampling the preprocessed data and building several smaller, simpler decision trees. Some benefits of using a random forest model include how robust it is against overfitting of the data because all of the weak learners are trained on different pieces of the data, it can be used to rank the importance of input variables, it is robust to outliers and nonlinear data, and it can run efficiently on large datasets.
The relu, tanh and sigmoid activations yielded a 74.12% accuracy, which is the best the model could produce using various number of neurons and layers. The next step should be to try the random forest classifier as it is less influenced by outliers.
