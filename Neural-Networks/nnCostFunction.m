function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== MAIN CODE HERE ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.
% Part 3: Implement regularization with the cost function and gradients.

% mapping of y
I=eye(num_labels);
Y=zeros(m,num_labels);
for i=1:m
	Y(i,:)=I(y(i),:);
end;

%================================ Part 1 ==============================%
% Forward Propagation
X=[ones(m,1) X];
a1=X;				% a1(5000,401)

z2 = a1 * Theta1';	% z2(5000,25)
a2=sigmoid(z2);		% a2(5000,25)
a2=[ones(m,1) a2];	% a2(5000,26)

z3 = a2 * Theta2';	% z3(5000,10)
a3 = sigmoid(z3);	% a3(5000,10)

h = a3;

% Cost
J = sum(sum((-Y .* log(h))-((1-Y) .* log(1-h)))) / m ;

Theta1Reg = Theta1(:, 2:size(Theta1,2)); % removing bias Theta1Reg(25,400)
Theta2Reg = Theta2(:, 2:size(Theta2,2)); % ThetaReg(10,25)

reg = (lambda/(2*m)) * ( sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)) );
J = J+reg;

%================================ Part 2 ==============================%

d3 = a3-Y;		%d3(5000,10)
d2 = (d3*Theta2)(:, 2:end).*sigmoidGradient(z2);		%d2(5000,25)

D2 = (1/m)*(d3' * a2);		% D2(10,26);without regularization term
D1 = (1/m)*(d2' * a1);		% D1(25,401) without regularization term

Theta1_grad = D1;
Theta2_grad = D2;


%================================ Part 3 ==============================%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+ (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+ (lambda/m)*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end;
