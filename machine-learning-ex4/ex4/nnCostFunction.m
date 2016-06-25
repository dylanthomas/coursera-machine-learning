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




% Part 1

X = [ones(m, 1) X]; % 5000*401

A1 = X' ; % 401%5000

Z2 = Theta1*A1; % 25*5000

A2 = sigmoid(Z2);
             
A2 = [ones(1,m) ; A2]; %26*5000

Z3 = Theta2*A2; %10*5000
             
A3 = sigmoid(Z3); % 10*5000
             

i = [ 1: m];
Y = sparse(i,y,1)'; % 10*5000
             
             

J = -1/m*sum(sum((Y.*log(A3) + (1 - Y).*log(1 - A3)))) + lambda/(2*m)*(sum(sumsq(Theta1(:, 2:end))) + sum(sumsq(Theta2(:, 2:end)))) ;
             
             
      
             
% Part 2

DEL2 = zeros(size(Theta2));
DEL1 = zeros(size(Theta1));

             
%Del3(:, i) = A3(:,i) - Y(:, i) ; % Del3 10*5000
%Del2(:, i) = (Theta2'*Del3(:, i))(2:end).*sigmoidGradient(Z2(:, i); % Del2 25*5000
%DEL2 = DEL2 + Del3(:, i)*A2(:,i)'; % 10*1  1*26 ==> 10*26
             
             
Del3 = A3 - Y ; % Del3 10*5000


               
Del2 = (Theta2'*Del3)(2:end, :).*sigmoidGradient(Z2); % Del2 25*5000
DEL2 = Del3*A2'; % 10*5000  5000*26 ==> 10*26
        
        
                              

DEL1 = Del2*X; % 25*5000 5000*401 ==> 25*401

               
               
             
             
% grad = 1/m*(sigmoid(X*theta) - y)'*X + lambda/m*[0, theta(2:end)'];




Theta2_grad = 1/m*DEL2 + lambda/m*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = 1/m*DEL1 + lambda/m*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)] ;
        









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
