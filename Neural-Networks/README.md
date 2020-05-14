## Classifier using Neural-Networks

Run the file **nn_classifier.m** with _X_train_, _y_train_, _X_test_ arguments.
This NN has only 1 hidden layer.

[Optional] you can give arguments like _lambda_, _input_layer_units_, _hidden_layer_units_, _num_labels_, _iterations_
like : 

` predictions= nn_classifier(X_tr, y_tr, X_test, lambda=1, input_layer_units=400, hidden_layer_units=25, num_labels=10, iterations=100)`
###### Note that given arguments values are by default

Here 
- _input_layer=400_ because for testing i use the dataset which have 20x20 size images
- _num_labels=10_ because the dataset 10 diffent digits to classify

### Testing
You can test this program via the dataset I provided in **Digit-Recognizer-With-Logi-Reg** folder.
Just download it and placed in the same folder where nn_classifier is downloaded.

Simply run these codes in Octave terminal:
```
load('data.mat');
X_train = X(1:4900, :);
y_train = y(1:4900, :);
X_test = X(4901:end, :);
y_test = y(4901:end, :);
pred= nn_classifier(X_train,y_train,X_test);
```

Also you can plot the data using **displayData.m** file , Just run `displayData(X)`


## Enjoy Guys . . 
