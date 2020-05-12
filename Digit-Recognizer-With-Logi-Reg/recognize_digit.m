function pred_digits= recognize_digit(X_train,y_train,X_test)

lambda= input('Enter Value of Lambda(Regularization param) : ');
K= input('Classes present in this data : ');

% 1. Training
[all_theta]= oneVsAll(X_train,y_train,K,lambda);


% 2. Evaluate
pred= predictOneVsAll(all_theta,X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred==y_train))*100);


% 2. Predict
pred_digits= predictOneVsAll(all_theta,X_test);

end;