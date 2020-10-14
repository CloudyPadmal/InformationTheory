%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      521390S - Information Theory                       %
%-------------------------------------------------------------------------%
%              Lab Assignment :: D.K.M.A.M.Padmal :: 2655246              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;
rng(100);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 10e3;                                        % Number of samples
mu = 0;                                          % Sample mean
sigma = 1;                                       % Sample variance
numBits = 8;                                     % Number of bits

% Generating normal random variables of size N with defined mu and sigma
Xs = normrnd(mu, sigma, [1, N]);                 % Sample data set (input)

distortion = 0:0.01:2;                           % Range of distortions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function verification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Function verification started ....................................');
% Verification of the @rateDistortion function %%%%%%%%%%%%%%%%%%%%%%%%%%%%
tryThis1 = 0;
if (tryThis1)
    disp('Rate distortion - Method 1 ...................................');
    R_D = rateDistortion(sigma, distortion, tryThis1);
end

% Verification of the @manualBlahuAri function %%%%%%%%%%%%%%%%%%%%%%%%%%%%
tryThis2 = 0;                                  
if (tryThis2)
    disp('Manual Blahut Arimoto algorithm - Method 2 ...................');
    % We need to run multiple iteration values to visualize the evolution
    % Iteration count is low (2)
    [~, Distortion_2, Rate_2] = manualBlahuAri(Xs, 2^numBits, 2);
    plot(Distortion_2, Rate_2); hold on;
    % Iteration count is medium (10)
    [~, Distortion_10, Rate_10] = manualBlahuAri(Xs, 2^numBits, 10);
    plot(Distortion_10, Rate_10); hold on;
    % Iteration count is high (50)
    [~, Distortion_50, Rate_50] = manualBlahuAri(Xs, 2^numBits, 50);
    plot(Distortion_50, Rate_50); 
    legend('Iteration (2)', 'Iteration (10)', 'Iteration (50)');
    title('Evolution of Blahut-Arimoto algorithm with increase of number of iterations');
    xlabel('Distortion (D)');
    ylabel('R(D)');
    hold off;
    pause;
end

% Verification of @probabilityMass function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tryThis3 = 0;
if (tryThis3)
    [X, DRr, RRr] = manualBlahuAri(Xs, 2^numBits, 2);
    disp('Evaluating PMF and CDF of Lloyd quantization of sample set ...');
    tProbMasses = zeros(1, length(X));
    tCumuDistri = zeros(1, length(X));

    for x = 1:length(X)
        tProbMasses(x) = probabilityMass(X, x, Xs);
        if (x > 2)
            tCumuDistri(x) = tCumuDistri(x - 1) + tProbMasses(x);
        end
    end

    subplot(1, 2, 1);
    stem(tProbMasses);
    title('Probability masses');
    xlabel('Quantization level');
    ylabel('Probability');
    xlim([1 2^numBits]);

    subplot(1, 2, 2); 
    plot(tCumuDistri);
    title('Cumulative probability')
    xlabel('Index count');
    ylabel('Probability');
    xlim([1 2^numBits]);
end

% Verification of @
disp('');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rate Distortion function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is defined for a gaussian source with a zero mean and sigma
% standard deviation. The output will only be available for distortions in
% the positive range upto the variance.
function R_D = rateDistortion(sigma, distortion, verification)

    % Filter out the invalid values
    filteredDistortion = (0 <= distortion) & (distortion < sigma);
    R_D = zeros(size(distortion));
    
    % Calculate the rate distortion
    R_Dt = 0.5 * log2(sigma ./ distortion(filteredDistortion));
    R_D(filteredDistortion) = R_Dt;
    
    % Plot for verification purposes
    if (verification)
        plot(distortion, R_D, 'k');
        title('Rate Distortion function R(D)');
        xlim([0 1]);
        ylim([0 6]);
        xlabel('Distortion (D)');
        ylabel('Rate Distortion R(D)');
        
        pause;
        
        plot(R_D, 10*log(distortion), 'k');
        title('Rate Distortion function R(D)');
        xlim([0 6]);
        ylabel('Distortion 10log_{10}(D)');
        xlabel('Rate Distortion R(D)');
        
        pause;
        
        plot(R_D, 10*log(distortion), 'k');
        title('Rate Distortion function R(D)');
        xlim([-0.02 0.1]);
        ylim([-3 7]);
        ylabel('Distortion 10log_{10}(D)');
        xlabel('Rate Distortion R(D)');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Manual Blahut-Arimoto function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is defined for a gaussian source with a zero mean and sigma
% standard deviation. The output will only be available for distortions in
% the positive range upto the variance.
function [Codes, Distortion, Rate] = ...
                                  manualBlahuAri(trainingSet, ...
                                                  codeBookSize, iterations)
    % Convert the training data set into a discrete set
    [~, Codes] = lloyds(trainingSet, codeBookSize);
    % Placeholder for distortion matrix
    distortionMatrix = zeros(codeBookSize, codeBookSize);
    % Calculate the distortion matrix
    for i = 1:codeBookSize
        for j = 1:codeBookSize
            distortionMatrix(i, j) = (Codes(i) - Codes(j)) ^ 2;
        end
    end
    
    % Calculate the probability mass values
    p_x = zeros(1, length(Codes));
    for x = 1:length(Codes)
        p_x(x) = probabilityMass(Codes, x, trainingSet);
    end
    
    % Define a range for the regularization parameter. Each value will
    % sweep along the distortion and rate curves until a match is met.
    lambdaRange = 0.001:0.1:20;
    
    % Placeholder for Distortion and Distortion rate values
    Distortion = zeros(1, length(lambdaRange));
    Rate = zeros(1, length(lambdaRange));
    % Placeholder to help with indices
    iterationID = 0;
    
    % Sweep for multiple lambda values
    for lambda = lambdaRange
        iterationID = iterationID + 1;
        
        % Initiate r_x as an element from a uniform distribution
        r_x = (1 / 256) * ones(1, codeBookSize);
        % Initiate q_x as a 256x256 null matrix
        q_x = zeros(codeBookSize, codeBookSize);
        
        % We try to iterate as much iterations as possible until the
        % conditional and reproduction probabilities do not change anymore.
        % Empirically the iteration count was sufficient to be 10 in this.
        for iter = 1:iterations
            % q_x is a 256x256 matrix. We would represent rows as x_hat and
            % columns as x_nrm and the same convention is used everywhere.
            
            % Calculating the conditional probability
            q_x_denominator = r_x * exp(-lambda * distortionMatrix);
            % Assign values to placeholder matrix
            for x_hat = 1:codeBookSize
                for x_nrm = 1:codeBookSize
                   q_x(x_nrm, x_hat) = r_x(x_hat) * exp(-lambda * ...
                       distortionMatrix(x_hat, x_nrm)) / ...
                       q_x_denominator(x_nrm);
                end
            end
            
            % Calculating the reproduction probability
            r_x = p_x * q_x;
            % Placeholder for return values
            RateDistortion = 0; DistortionRate = 0;
            % Estimate the values for optimization constraints
            for x_hat = 1:codeBookSize
                for x_nrm = 1:codeBookSize
                    % Since matlab does not interpret 0log0 as 0, we have
                    % to discard the q values that are too small.
                    if (q_x(x_nrm, x_hat) < realmin)
                        continue;
                    end
                    RateDistortion = RateDistortion + ...
                        (p_x(x_nrm) * q_x(x_nrm, x_hat) * ...
                        log2(q_x(x_nrm, x_hat) / r_x(x_hat)));
                    DistortionRate = DistortionRate + ...
                        (p_x(x_nrm) * q_x(x_nrm, x_hat) * ...
                        distortionMatrix(x_nrm, x_hat));
                end
            end
            Distortion(iterationID) = DistortionRate;
            Rate(iterationID) = RateDistortion;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Probability Mass function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function calculates the probability mass function for a set of codes
% quantized using lloyd for a given number of bits.
function PMF = probabilityMass(codes, index, sampleSet)
    % Consider the corner cases
    if (index == 1)
        intervalBreak = (codes(index + 1) + codes(index)) / 2;
        PMF = sum(sampleSet > -inf & sampleSet <= intervalBreak);
    elseif (index == length(codes))
        intervalBreak = (codes(index - 1) + codes(index)) / 2;
        PMF = sum(sampleSet > intervalBreak & sampleSet < inf);
    else
        intervalBreak1 = (codes(index + 1) + codes(index)) / 2;
        intervalBreak2 = (codes(index - 1) + codes(index)) / 2;
        PMF = sum(sampleSet <= intervalBreak1 & ...
                    sampleSet > intervalBreak2);
    end
    PMF = PMF / length(sampleSet);
end
