%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA108
% Project Title: Population Size Adaptation Covariance Matrix Adaptation 
%                 Evolution Strategy (PSA-CMA-ES)
% Developer: Pedro Bandeira
% 
% Implementation of PSA-CMA-ES as described by Kouhei Nishida and Youhei
% Akimoto on their paper.
%
% Reference:
% Kouhei Nishida and Youhei Akimoto. 2018. PSA-CMA-ES: CMA-ES with 
% population size adaptation. In Proceedings of the Genetic and 
% Evolutionary Computation Conference (GECCO '18), Hernan Aguirre (Ed.). 
% ACM, New York, NY, USA, 865-872. 
% DOI: https://doi.org/10.1145/3205455.3205467 
%
% Code based on CMAES from Yarpiz. Please give it a look for more
% information.
% 

clc;
clear;
close all;

dimensions = [10 30];
f_to_eval = [1 2 6 7 9 14];
dim = 10;

%for dim = dimensions
    %for function_to_eval = f_to_eval:
        %% Problem Settings

        %CostFunction=@(x)cec14_func(x',1);   % Cost Function
        CostFunction=@Ackley;
        
        dim;                % Number of Unknown (Decision) Variables

        VarSize=[1 dim];       % Decision Variables Matrix Size

        VarMin=-100;             % Lower Bound of Decision Variables
        VarMax= 100;             % Upper Bound of Decision Variables

        %% CMA-ES Settings

        % Population Size (and Number of Offsprings)
        lambda_default=(4+round(3*log(dim)));
        
        % Sigma0
        sigma0=0.3*(VarMax-VarMin);
        
        %PSA Params
        alpha_ = 1.4;
        beta_ = 0.4;
        lambda = lambda_default;
        lambda_r = lambda_default;
        lambda_min = lambda_default;
        lambda_max = 200;
        
        % Maximum Number of Iterations
        MaxEvals = 10000*dim;
        MaxIt=round(MaxEvals/lambda_min);        
        nb_evals = 0;
        % Number of Parents
        mu=round(lambda_default/2);

        % Parent Weights
        w=log(mu+0.5)-log(1:mu);
        w=w/sum(w);
        w=[w,zeros(1,lambda-mu)];
        
        % Number of Effective Solutions
        mu_eff=1/sum(w.^2);

        % Step Size Control Parameters (c_sigma and d_sigma);
        %sigma0=0.3*(VarMax-VarMin);
        cs=(mu_eff+2)/(dim+mu_eff+5);
        ds=1+cs+2*max(sqrt((mu_eff-1)/(dim+1))-1,0);
        qsi_n=sqrt(dim)*(1-1/(4*dim)+1/(21*dim^2));

        % Covariance Update Parameters
        cc=(4+mu_eff/dim)/(4+dim+2*mu_eff/dim);
        c1=2/((dim+1.3)^2+mu_eff);
        alpha_mu=2;
        cmu=min(1-c1,alpha_mu*(mu_eff-2+1/mu_eff)/((dim+2)^2+alpha_mu*mu_eff/2));
        hth=(1.4+2/(dim+1))*qsi_n;

        %% Initialization

        ps=cell(MaxIt,1);
        pc=cell(MaxIt,1);
        p_theta = cell(MaxIt,1);
        gamma_s = cell(MaxIt,1);
        gamma_c = cell(MaxIt,1);
        gamma_theta = cell(MaxIt,1);
        C=cell(MaxIt,1);
        sigma=cell(MaxIt,1);

        ps{1}=zeros(VarSize);
        pc{1}=zeros(VarSize);
        %p_theta{1} = zeros(VarSize);
        gamma_s{1} = 0;
        gamma_c{1} = 0;
        gamma_theta{1} = 0;
        C{1}=eye(dim);
        sigma{1}=sigma0;

        empty_individual.Position=[];
        empty_individual.Step=[];
        empty_individual.Cost=[];

        M=repmat(empty_individual,MaxIt,1);
        M(1).Position=unifrnd(VarMin,VarMax,VarSize);
        M(1).Step=zeros(VarSize);
        M(1).Cost=CostFunction(M(1).Position);

        BestSol=M(1);

        BestCost=zeros(MaxIt,1);
        

  
        %% CMA-ES Main Loop

        for g=1:MaxIt

            % Generate Samples
            pop=repmat(empty_individual,lambda_max,1);
            for i=1:lambda
                pop(i).Step=mvnrnd(zeros(VarSize),C{g});
                pop(i).Position=M(g).Position+sigma{g}*pop(i).Step;
                pop(i).Cost=CostFunction(pop(i).Position);
                nb_evals = nb_evals + 1;
                % Update Best Solution Ever Found
                if pop(i).Cost<BestSol.Cost
                    BestSol=pop(i);
                end
            end

            % Sort Population
            Costs=[pop.Cost];
            [Costs, SortOrder]=sort(Costs);
            pop=pop(SortOrder);

            % Save Results
            BestCost(g)=BestSol.Cost;

            % Display Results
            if mod(g, 10) == 0
                disp(['Iteration ' num2str(g) ': Best Cost = ' num2str(BestCost(g))]);
            end
            
            % Exit if no further improvement
            %if abs(BestCost(g) - function_to_eval*100) < 1e-8
            %    break;
            %end
            % Exit At Last Iteration
            if g==MaxIt
                break;
            end

            % Update Mean
            M(g+1).Step=0;
            M(g+1).Cost=0;
            for j=1:mu
                M(g+1).Step=M(g+1).Step+w(j)*pop(j).Step;
            end
            M(g+1).Position=M(g).Position+sigma{g}*M(g+1).Step;
            %M(g+1).Cost=CostFunction(M(g+1).Position);
            %if M(g+1).Cost<BestSol.Cost
            %    BestSol=M(g+1);
            %end

            % Update Step Size
            ps{g+1}=(1-cs)*ps{g}+sqrt(cs*(2-cs)*mu_eff)*M(g+1).Step/chol(C{g})';
            % gamma_s
            gamma_s{g+1} = (1-cs)^2 *gamma_s{g} + cs*(2-cs);

            sigma{g+1}=sigma{g}*exp(cs/ds*(norm(ps{g+1})/qsi_n-1))^0.3;

            % Update Covariance Matrix
            if norm(ps{g+1})/sqrt(1-(1-cs)^(2*(g+1)))<hth
                hs=1;
            else
                hs=0;
            end
            delta=(1-hs)*cc*(2-cc);
            pc{g+1}=(1-cc)*pc{g}+hs*sqrt(cc*(2-cc)*mu_eff)*M(g+1).Step;
            % gamma_c
            gamma_c{g+1} = (1-cc)^2*gamma_c{g}+hs*cc*(2-cc);

            C{g+1}=(1-c1-cmu)*C{g}+c1*(pc{g+1}'*pc{g+1}+delta*C{g});
            for j=1:mu
                C{g+1}=C{g+1}+cmu*w(j)*pop(j).Step'*pop(j).Step;
            end

            % If Covariance Matrix is not Positive Defenite or Near Singular
            [V, E]=eig(C{g+1});
            if any(diag(E)<0)
                E=max(E,0);
                C{g+1}=V*E/V;
            end
            
            %% Update evolution path and its factor

            % Fisher Information Matrix
            FIM_sqrt = sqrtm(ecmnfish(reshape([pop.Position], lambda, dim), C{g+1}));
            % Delta theta
            Sigma = C{g+1}./sigma{g+1};
            trilSigma = tril(Sigma);
            vechSigma = trilSigma(trilSigma~=0);
            d_theta = [M(g+1).Position, vechSigma']';
            % Expected value of Fisher Information Matrix
            first_bracket = 1 + 8 * gamma_s{g+1} * (dim - qsi_n^2)/qsi_n^2 * (cs/ds)^2;
            second_bracket = (dim^2+dim)*cmu^2/mu_eff + (dim^2+dim)*cc*(2-cc)*c1*cmu*mu_eff*sum(w.^3) + c1^2*(gamma_c{g+1}^2*dim^2 + (1-2*gamma_c{g+1}+2*gamma_c{g+1}^2)*dim);
            expected_value_fisher_matrix = (dim * cmu^2)/mu_eff + (2*dim*(dim-qsi_n^2)/qsi_n^2)*gamma_s{g+1}*(cs/ds)^2 + 0.5*first_bracket*second_bracket;
            % p_theta evolution
            if isempty(p_theta{1})
                p_theta{1} = zeros(size(d_theta));
            end
            p_theta{g+1} = (1 - beta_).*p_theta{g} + sqrt(beta_*(2-beta_)) * (FIM_sqrt * d_theta)/sqrt(expected_value_fisher_matrix);
            gamma_theta{g+1} = (1-beta_)^2*gamma_theta{g}+beta_*(2-beta_);
            
            % lambda evolution
            lambda_r = lambda_r * exp(beta_*(gamma_theta{g+1} - norm(p_theta{g+1})^2/alpha_));
            lambda_r = min(max(lambda_r, lambda_min), lambda_max);
            expected_value_normal_order = mean(reshape([pop(1:lambda).Position], lambda, dim), 2);
            weighted_average_from_lambda_old = -sum(w.*expected_value_normal_order');
            lambda = round(lambda_r);
            
            % Exit if next iteration exceeds nb_evals
            if nb_evals + lambda > MaxEvals
                break;
            end
            
            % step-size correction
            expected_value_normal_order = mean(reshape([pop(1:lambda).Position], lambda, dim), 2);
            weighted_average_from_lambda = -sum(w.*expected_value_normal_order');
            scaling_factor_new = weighted_average_from_lambda * dim * mu_eff/(dim - 1 + weighted_average_from_lambda^2 * mu_eff);
            scaling_factor_old = weighted_average_from_lambda_old * dim * mu_eff/(dim - 1 + weighted_average_from_lambda_old^2 * mu_eff);
            sigma{g+1} = sigma{g+1} * (scaling_factor_new/scaling_factor_old);
            
        end
    %end
%end
%% Display Results

figure;
% plot(BestCost, 'LineWidth', 2);
semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

