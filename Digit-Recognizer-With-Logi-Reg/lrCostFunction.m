function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ====================== MAIN CODE HERE ======================

h=sigmoid(X*theta);
J = ( -(y' * log(h)) - ((1-y)' * log(1-h)) ) / m; 
regu_param = (lambda/(2*m)) * sum(theta( 2:size(theta) ).^2) ;
J = J + regu_param;
grad = ( (X'*(h-y))/m );
temp=theta;
temp(1) = 0;
grad = grad + (lambda .* temp)/m;
grad(1,:)=(X(:,1)' * (h-y)) / m ;


% =============================================================

grad = grad(:);

end
