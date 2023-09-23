%=============================
% Author: Antonis Kastellakis
%=============================
% Learning hyperparameters alpha and beta from the data  
% by maximizing the Evidence Function.
function [alpha_opt, beta_opt] = parameter_optimization(Phi,t,alpha,beta)
    
    % Because the matrix Phi^T*Phi is fixed, we can compute its eigenvalues once
    % at the start and then simply multiply these by beta to obtain the lambdas.
    M = size(Phi,2);
    N = size(Phi,1);
    eigenvals = eig((Phi') * Phi);
    % Use the arbitrary values of alpha and beta and iterate 
    % until convergence
    old_alpha = alpha;
    old_beta = beta;
    % Iterative optimization
    while (1)
        %step I: given alpha and beta, compute gamma and mN
        lambda = old_beta * eigenvals;
        alpha_plus_lambda = old_alpha + lambda;
        gamma = sum( lambda ./ alpha_plus_lambda );
        SN_inv = old_alpha * eye(M) + old_beta * (Phi') * Phi;
        SN = inv( SN_inv );
        mN = old_beta * SN * ( Phi' )* t;
        %step II: given gamma and mN, compute alpha and beta
        new_alpha = gamma / (mN' * mN);
        beta_sum = 0;
        for n=1:N
            phi_xn = Phi(n,:)';
            beta_sum = beta_sum + ( t(n) - (mN') * phi_xn )^2;
        end
        new_beta_inv = beta_sum / (N - gamma);
        new_beta = 1 / new_beta_inv;
        if ( ( abs(new_alpha - old_alpha) < 1e-3 ) && ( abs(new_beta - old_beta) < 1e-3 ) )
            break
        else
            old_alpha = new_alpha;
            old_beta = new_beta;
        end
    end
    % After convergence we have the optimal parameters
    alpha_opt = new_alpha;
    beta_opt = new_beta;
end
