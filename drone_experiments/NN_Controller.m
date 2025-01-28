clear all
close all
clc

global SIASclient    FLAG
global goalDesired Error CommandInput SS SS2 commandArray_Idf Cont GoalPt CI
global deltaT inc drone_mode ContGoal xyDesired vertDesired yawDesired DESIREDPOINT
global p state state2 xyDesired vertDesired rollDesired pitchDesired yawDesired DESIRED
global SIASclient Error CommandInput ContGoal Time netx nety netz
global goalDesired SS SS2 commandArray_Idf CI
global deltaT drone_mode Cont GoalPt DESIRED1
global p state state2 inc goalDesired deltaT  xyDesired vertDesired rollDesired pitchDesired yawDesired DESIREDPOINT
 

%% LOAD NN MODELS
% Load ONNX model--X
netx = importNetworkFromONNX("weights_drone_weight_best/1_x/NN_x.onnx");
% model = importONNXNetwork('model.onnx');

netx.Layers(1).InputInformation;

X = dlarray(rand(1, 3), 'UU');
netx = initialize(netx, X);
summary(netx)


% Load ONNX model--Y
nety = importNetworkFromONNX("weights_drone_weight_best/1_y/NN_y.onnx");
% model = importONNXNetwork('model.onnx');

nety.Layers(1).InputInformation;

nety = initialize(nety, X);
summary(nety)

% Load ONNX model--X
netz = importNetworkFromONNX("weights_drone_weight_best/1_z/NN_z.onnx");
% model = importONNXNetwork('model.onnx');

netz.Layers(1).InputInformation;

netz = initialize(netz, X);
summary(netz)





%%

SIASclient = natnet;
SIASclient.connect; 
pause(2)

fprintf('\n\nConnecting to Drone...\n') 
p = parrot(); 
fprintf('Connected to %s\n', p.ID) 
fprintf('Battery Level is %d%%\n', p.BatteryLevel)
takeoff(p); 
pause(1)







 
 Error=zeros(4,1);
 CommandInput=zeros(4,1);
inc=1;


while(1>0)


Time(inc,1)=double(SIASclient.getFrame.fTimestamp);

    Position=double([SIASclient.getFrame.RigidBodies(1).x;SIASclient.getFrame.RigidBodies(1).y;SIASclient.getFrame.RigidBodies(1).z]);
    q=quaternion( SIASclient.getFrame.RigidBodies(1).qw, SIASclient.getFrame.RigidBodies(1).qx, SIASclient.getFrame.RigidBodies(1).qy, SIASclient.getFrame.RigidBodies(1).qz );
    eulerAngles=quat2eul(q,'xyz')*180/pi;
    Angle=[eulerAngles(1);eulerAngles(2);eulerAngles(3)];
    state=[Position;Angle];
    [errorArray]=ControlCommand;
    if FLAG(inc)==1
        SS(:,inc)=state;
        CI(:,inc)=commandArray_Idf;
        Error(:,inc)=errorArray;
        inc=inc+1;
    end
% end

end


function [errorArray]=ControlCommand
 
global p state  inc FLAG SS CI Error commandArray_Idf  GoalPt ContGoal goalDesired  Time DESIRED netx nety netz


%% Define desired tolerances and gains



 if inc==1
     dt=0.008301;
 else
     dt=Time(inc,1)-Time(inc-1,1);
 end

if dt < eps 
   dt = 0.008301; % Average calculated time step
end


if inc==1
    example_x = dlarray([state(1),(state(1))/dt,0], 'UU');
else
example_x = dlarray([state(1),(state(1)-SS(1,inc-1))/dt,0], 'UU');
end

% Perform inference
output_x = predict(netx, example_x);
ux = extractdata(output_x(4));


if ux>=0.05
    ux=0.05;
elseif ux<=-0.05
    ux=-0.05;
end

if inc==1
    example_y = dlarray([state(2),(state(2))/dt,0], 'UU');
else
    example_y = dlarray([state(2),(state(2)-SS(2,inc-1))/dt,0], 'UU');
end

% Perform inference
output_y = predict(nety, example_y);
uy = extractdata(output_y(4));

if uy>=0.05
    uy=0.05;
elseif uy<=-0.05
    uy=-0.05;
end


if inc==1
    example_z = dlarray([state(3),(state(3))/dt,1.5], 'UU');
else
    example_z = dlarray([state(3),(state(3)-SS(3,inc-1))/dt,1.5], 'UU');
end

% Perform inference
output_z = predict(netz, example_z);
uz = extractdata(output_z(4));

if uz>=0.5
    uz=0.5;
elseif uz<=-0.5
    uz=-0.5;
end


 if inc == 1 


 old_yaw_Error = 0; 


 else
     old_yaw_Error=Error(1,inc-1);

 end


 yawActual = deg2rad(state(6)); 


 
 % to rotate X,Y from world frame to robot frame
 Tw2r = [cos(yawActual), sin(yawActual); -sin(yawActual), cos(yawActual)]; 


 % 
 
 
 
 % Compute the errors
 % Yaw Error
 if inc==1
     dt=0.008301;
 else
     dt=Time(inc,1)-Time(inc-1,1);
 end

if dt < eps 
   dt = 0.008301; % Average calculated time step
end

% dt

yawe1=(deg2rad(0) - yawActual);
 yawError = wrapToPi(yawe1);
 yawD_Error = (yawError-old_yaw_Error)/dt;
 
kYaw =-1; 
kD_Yaw =-0.8;

 % compute the yaw commands
 yawCmd = kYaw*yawError+kD_Yaw*yawD_Error;
 
 if abs(yawCmd) > 3.4 
 yawCmd = sign(yawCmd)*3.4; 
 end 


 

 errorArray = [yawError]; 


  if inc>1 
      if Time(inc,1)-Time(inc-1,1)>eps
          commandArray_Idf= [ux; uy; yawCmd; uz];
          FLAG(inc)=1;
  else
      commandArray_Idf= [CI(1,inc-1); CI(2,inc-1); CI(3,inc-1); CI(4,inc-1)];
      FLAG(inc)=0;
      end
  else
      commandArray_Idf= [ux; uy; yawCmd; uz];
       FLAG(inc)=1;
  end


move(p, 0.1, 'RotationSpeed', commandArray_Idf(3),'VerticalSpeed', commandArray_Idf(4),'roll', commandArray_Idf(2), 'pitch', commandArray_Idf(1));







end


