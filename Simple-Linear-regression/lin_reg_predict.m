function [y_predicted] = lin_reg_predict(X_train,y_train,X_test)

% Computes the predicted value of y if training features and labels are provided
%---Your Comment Goes here-----

% 1. Feature Normalize
X_train = featureNormalize(X_train);
y_train = featureNormalize(y_train);
%X_test = featureNormalize(X_test);
%y_test = featureNormalize(y_test);

% 2. Getting the Theta
if (size(X_train,1) < 10)
	theta = normalEqn(X_train,y_train);
	J = computeCostMulti(X_train,y_train,theta);
	disp('Total Mean Square Error on Training set : '),disp(J);
else 
	theta=zeros(size(X_train,2),1);		% X_train(m,n) theta(n,1)
	fflush(stdout);
	lr=input('Enter Learning Rate for GradientDescent : ');
	fflush(stdout);
	iters=input('Enter iterations for GradientDescent : ');
	[theta,costHist] = gradientDescentMulti(X_train,y_train,theta,lr,iters);
	disp('Size of Mean Square Error: '),disp(size(costHist));
	cho=input('Want to see last 5 Errors ? (1/0) ');
	if (cho==1)
		disp(costHist( size(costHist,1)-5 : end, :));
	endif;
endif;

% 3. Making Prediction
y_predicted = zeros(size(X_test,1),1);
y_predicted = X_test*theta;



