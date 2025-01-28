%% This is the new system: Inverted Pedulum system

clc
clear all


%%  Basic Settings
% theta
theta = 0.01;
% Save the data to a text file
filename = 'IP_roi_001theta_ref.csv';


%% Loop starts

% [column1:x1, x2 col2-3:P, col4: K]

count = 1;
for r = -1:0.1:1
    % reference angle
    for x1 = -5:0.1:5
        % init degrees
        for x2 = -5:0.1:5
            % init ang. velocity

            x = [x1*pi/180;x2*pi/180];
            x_r = r*pi/180;

            % variables to optimize
            u = sdpvar(1,1,'full');
            P=sdpvar(2,2,'symmetric');

            Bd=[0;0.1];
            Ad=[1 0.1;-0.1*9.8*cos(x1) 1];
            
            Objective = 0.1*(norm(u))^2+ 2*(norm(Ad*x+Bd*u-x_r)^2)+((x-x_r)'*P*(x-x_r));
            Constraints = [P>=0.001;
            ((x-x_r)'*P*(x-x_r))>=((1.5*theta)^2)*(x-x_r)'*(x-x_r);
            ((Ad*x+Bd*u-x_r)'*P*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
        
            % opt=sdpsettings('solver','bmibnb');
            opt=sdpsettings('solver','fmincon','MaxIter',2000);
            sol=optimize(Constraints,Objective,opt);
                    
            
            chache_matrix(:,1) = x;
            p = double(P);
            % eig(p)
            chache_matrix(:,2:3) = p;
            chache_matrix(1,4) = double(u);
            chache_matrix(2,4) = x_r;
            result_matrix(:,:,count) = chache_matrix;
            count = count+1;
        
                
            % Clear variables
            clear('yalmip')
    
        end
    end
    % save result matrix and clear it. recount numbers
    writematrix(result_matrix, filename,'WriteMode','append');
    clear result_matrix
    count = 1;

end



