function predicts = admitted_or_not(X,y,X_test)
%clear;clc;


% 1. Compute Theta
[m,n] = size(X);
X = [ones(m,1) X];
theta= zeros(n+1,1);
options= optimset('GradObj', 'on', 'MaxIter', 500);
[theta,cost]= fminunc(@(t)(costFunction(t,X,y)), theta, options);
fprintf('Cost is : %f\n', cost);


% 2. Evaluating on Traning Set
p=predict(theta,X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

disp(' ');
% 3. Make Prediction
X_test= [ones(size(X_test,1),1) X_test];
y_preds = predict(theta,X_test);
prob=sigmoid(X_test*theta);
if(size(X_test,1) < 2)
	if(y_preds==0)
		disp('Not Admitted !');
		fprintf('Probability is : %f\n',prob);
		predicts=0;
	else
		disp('Admitted !');
		fprintf('Probability is : %f\n',prob);
		predicts=1;
	endif;
else
predicts=y_preds;
predicts= cellstr(predicts);
for i=find(y_preds==0)
	predicts(i)= "Not Admitted";
endfor;

for i=find(y_preds==1)
	predicts(i)= "Admitted";
endfor;

endif;

disp(' ');
disp(' ');

end;