# Neural Network Regression with TensorFlow on Insurance Data

## Project Overview

This project focuses on using a neural network regression model built with TensorFlow to predict insurance costs based on various customer attributes. The goal is to develop a predictive model that accurately estimates insurance charges, helping insurance companies to set premiums and understand the risk factors associated with different customers.

## Dataset

- **Source:** The dataset includes customer information such as age, gender, BMI, number of children, smoking status, and region, along with the insurance charges (target variable).
- **Features:** Key features include demographic data, lifestyle choices (e.g., smoking), and regional information.

## Tools & Libraries Used

- **Data Analysis:**
  - `Pandas` for data manipulation and analysis
  - `Matplotlib` and `Seaborn` for data visualization
- **Neural Network Development:**
  - `TensorFlow` and `Keras` for building and training the neural network model
- **Model Evaluation:**
  - Metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to evaluate the performance of the regression model

## Methodology

1. **Data Exploration:**
   - Conducted exploratory data analysis (EDA) to understand the distribution of features and their relationship with insurance charges.

2. **Data Preprocessing:**
   - Handled missing values, normalized numerical features, and encoded categorical variables to prepare the data for the neural network model.

3. **Model Development:**
   - Built a neural network regression model using TensorFlow and Keras. The architecture included multiple layers with activation functions like ReLU to capture complex relationships between the features and the target variable.
   - Used a combination of dropout layers and batch normalization to prevent overfitting and improve model generalization.

4. **Model Training:**
   - Trained the neural network on the preprocessed data, using a loss function appropriate for regression (e.g., MSE) and optimizing the model with an Adam optimizer.
   - Monitored the model's performance using validation data to ensure it was not overfitting.

5. **Model Evaluation:**
   - Evaluated the model's performance on a test set using metrics like MSE and MAE to determine how well the model predicted insurance charges.
   - Visualized the model's predictions against actual insurance charges to assess its accuracy.


## Results

- The neural network model was able to predict insurance charges with a reasonable degree of accuracy, showing the potential of deep learning in regression tasks.
- Key features such as age, BMI, and smoking status were identified as significant predictors of insurance costs.

## Conclusion

This project successfully demonstrated the use of a neural network regression model to predict insurance costs. The results indicate that deep learning models can effectively capture complex relationships in data, providing valuable insights for pricing insurance premiums.

## Future Work

- Experiment with different neural network architectures and hyperparameter tuning to further improve model performance.
- Incorporate additional features, such as medical history or lifestyle factors, to enhance the predictive power of the model.
- Deploy the model in a real-world insurance application to provide dynamic pricing based on customer profiles.

