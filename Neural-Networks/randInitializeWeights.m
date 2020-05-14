function W = randInitializeWeights(L_in, L_out)
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%

W = zeros(L_out, 1 + L_in);

% ====================== MAIN CODE HERE ======================

epsilon_init = 0.12;
W = rand(L_out, 1+L_in) * 2*epsilon_init - epsilon_init;


% =========================================================================

end
