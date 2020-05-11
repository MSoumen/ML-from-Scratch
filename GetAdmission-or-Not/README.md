## GetAdmission-or-Not

Run __admitted_or_not.m__ with X_train,y_train,X_test arguments.

_**It shows if a Student can admitted in a college or not**_.

Where 
- X_train is Features of trainning exampls
- y_train is Labels of trainning exampls
- X_test is the features you give to predict

To test this program , I provided a data file named __data1.txt__.
load the dataset using.
```Octave
   data = load('data1.txt'),
   X_train = data(:,1:2), 
   y_train = data(:,3)
```
   
Also you can plot the data using :.

`plotData(X,y)` .

and Plot the decision boundary using :.

`plotDecisionBoundary(theta,X,y)`
