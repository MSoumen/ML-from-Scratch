## Digit-Recognizer-With-Logistic_Regression

Run __recognize_digit.m__ with X_train,y_train,X_test arguments.

_**It shows what is the number, from a image's pixel values(20x20=400)**_.

Where 
- X_train(m x 400) is Features of trainning exampls
- y_train is Labels of trainning exampls
- X_test is the array of pixel values you give to predict

To test this program , I provided a data file named __data.mat__.
load the dataset and set values of X_train,y_train using.
```Octave
   load('data.mat');
   X_train = X, 
   y_train = y;
```
   
Also you can see the image from its pixel values using :.

`displayData(X)` .
