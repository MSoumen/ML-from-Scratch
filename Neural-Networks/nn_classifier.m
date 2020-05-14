function pred= nn_classifier(X_train,y_train,X_test, lambda, input_layer_units, hidden_layer_units, num_labels, iterations)

% Set example_width automatically if not passed in
if ~exist('lambda', 'var') || isempty(lambda) 
	lambda = 1;
end;
if ~exist('input_layer_unit', 'var') || isempty(input_layer_units) 
	input_layer_units=400;
end;
if ~exist('hidden_layer_unit', 'var') || isempty(hidden_layer_units) 
	hidden_layer_units=25;
end;
if ~exist('num_labels', 'var') || isempty(num_labels) 
	num_labels = 10;
end;
if ~exist('iterations', 'var') || isempty(iterations) 
	iterations = 100;
end;



[m,n]= size(X_train);

% Initialize Thetas
theta1= randInitializeWeights(input_layer_units,hidden_layer_units); %takes(layer_in, layer_out)
theta2= randInitializeWeights(hidden_layer_units,num_labels);

%unrolling params
ini_nn_params= [theta1(:); theta2(:)];

% Training  NN
opts= optimset('MaxIter', iterations);
costFunc= @(p) nnCostFunction(p, input_layer_units, hidden_layer_units, num_labels, X_train,y_train, lambda);

[nn_params, cost]= fmincg(costFunc, ini_nn_params, opts);

% Obtaining Thetas from nn_params
theta1= reshape(nn_params(1:hidden_layer_units*(input_layer_units+1)), hidden_layer_units, (input_layer_units+1));
theta2= reshape(nn_params((1+(hidden_layer_units*(input_layer_units+1))): end), num_labels, (hidden_layer_units+1));


% Evaluation
p=predict(theta1,theta2,X_train);
fprintf("\nTraining set Accuracy: %f\n", mean(double(p==y_train))*100);


% Prediction
pred= predict(theta1,theta2,X_test);

end;


