function [rank] = LRMFS(X,Y,options)

[nFeat,nSamp] = size(X);
[~,nLabel]=size(Y);

W = rand(nFeat,nLabel);
F= rand(nSamp,nLabel);
Q=rand(nFeat,nLabel);
B=Construct_B(Y);
S = rand(nSamp,nLabel);
I =  eye(nFeat);

para.k=20;%size(temp_target,1) - 1;
Lx = Laplacian_GK(X',para.k);
para.k=size(Y,2)-1;
Ly = Laplacian_GK1(Y,para);

D=eye(nFeat);
Dz = eye(nSamp);

alpha=options.alpha;
beta=options.beta;
gamma=options.gamma;
lambda=options.lambda;

Niter=11; %10
err=1;
iter=1;
while (err > 10^-3 & iter< Niter)
    
F=Y+B.*S;

%% A    
A = Q'*W;    

%% W
C=X*Dz*X'+alpha*I+lambda*D;
G=C-alpha*Q*Q';
W=inv(G)*X*Dz*F;

%% Q
P=inv(I-alpha*C)*inv(C)*X*Dz*F*F'*Dz*X'*inv(C);
U1=2*beta*Lx-alpha*P-alpha*P';
U2=2*gamma*Ly;
Z=zeros(nFeat,nLabel);
Q=lyap(U1, U2, Z);

%% S
S_original=B.*(X'*W-Y);
[x11,y11]=size(Y);
for i=1:x11
for j=1:y11
    S(i,j)=max(S_original(i,j),0);
end
end  

%% Dz,D
V=X'*W-F;
Wi = sqrt(sum(V.*V,2));
d = 0.5./(Wi+eps);
Dz = diag(d);

wi = sqrt(sum(W.*W,2));
d = 0.5./(wi+eps);
D = diag(d);

obj(iter)= trace(V'*Dz*V)+alpha*norm(W-Q*A,'fro')^2+beta*trace(Q'*Lx*Q)+gamma*trace(Q*Ly*Q')+lambda*trace(W'*D*W);
if iter>1
        err = abs(obj(iter-1)-obj(iter));
end
iter=iter+1;
end
[~, rank] = sort(sum(W.*W, 2), 'descend');
end



function B=Construct_B(Y)
%%
[x1,y1]=size(Y);
B_origin=Y;
for i=1:x1
    for j=1:y1
      if Y(i,j)==0
        B_origin(i,j)=-1;
      end
    end
end
B=B_origin;
end