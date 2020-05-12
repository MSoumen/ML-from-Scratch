function p = predictOneVsAll(all_theta, X)
%The labels are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);  % all_theta(K,n+1)
p = zeros(size(X, 1), 1); % p(m,1)

% Add ones to the X data matrix
X = [ones(m, 1) X]; % X(m,n+1)

% ====================== MAIN CODE HERE ======================



pred=sigmoid(X * all_theta');
[pred_max, index_max] = max(pred, [], 2);  % index_max will give the col which have the maximum value
p=index_max;




% =========================================================================


end;
