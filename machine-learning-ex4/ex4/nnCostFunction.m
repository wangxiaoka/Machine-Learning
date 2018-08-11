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
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];   % 5000 x 401
y_t = zeros(m,num_labels);  % 5000 x 10
c = 1:num_labels;
for i = 1:m
  y_t(i,:) = (y(i) == c);  % 5000 x 10
end

z2 = Theta1 * X';      % (25 x 401) x (401 x 5000)
a2_t = sigmoid(z2);    % 25 x 5000

a2 = [ones(1,size(a2_t,2)); a2_t];  % 26 x 5000
%size(a2)
z3 = Theta2 * a2;      % (10 x 26) x (26 x 5000)
a3_t = sigmoid(z3);    % 10 x 5000
a3 = a3_t';    % 5000 x 10;
%size(a3)

for i = 1:m
  J = J + y_t(i,:) * log(a3(i,:)') + ((1 - y_t(i,:)) * log(1 - a3(i,:)'));
end

J_Theta1 = Theta1(:,(2:end));
J_Theta2 = Theta2(:,(2:end));

J = (-1) * J / m + lambda*(sum(sum(J_Theta1.^2)) + sum(sum(J_Theta2.^2))) / (2*m);


D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

d3 = zeros(m,num_labels);

a1 = X;   % 5000 X 401

z2 = Theta1 * X';      % (25 x 401) x (401 x 5000)
a2_t = sigmoid(z2);    % 25 x 5000
a2 = [ones(1,size(a2_t,2)); a2_t];  % 26 x 5000
%size(a2)

z3 = Theta2 * a2;      % (10 x 26) x (26 x 5000)
a3_t = sigmoid(z3);    % 10 x 5000
a3 = a3_t';    % 5000 x 10;
%size(a3)

d3 = a3 - y_t;  % 5000 x 10
z2_t = [ones(1,size(z2,2));z2];   % 26 x 5000
d2 = Theta2' * d3' .* sigmoidGradient(z2_t);  % [(26 x 10) * (10 x 5000)] .* [26 x 5000] 


D1 = D1 + d2(2:end,:) * a1;  % [25 x 401] + [(25 x 5000) * (5000 x 401)]
D2 = D2 + d3' * a2';  % [10 x 26] + [(10 x 5000) * (5000 x 26)]



Theta1_grad = D1 / m;
Theta1_grad(:,2:end) = D1(:,2:end) / m + lambda * Theta1(:,2:end) / m;

Theta2_grad = D2 / m;
Theta2_grad(:,2:end) = D2(:,2:end) / m + lambda * Theta2(:,2:end) / m;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
