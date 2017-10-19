%-------------------------SAMPLE PREPARATION-----------------------------
%varnames=bosthous.Properties.VariableNames; 
%varnames(end)=[]; 
%bosthous=table2array(bosthous); 
%MEDV=bosthous(:,end); 
%bosthous(:,end)=[];
MEDV=MEDVres;
%MEDV=log10(MEDV+1);
curr_seed=rng; %Save current seed
my_seed=7; %Set custom seed 
sample_size=length(bosthous(:,1)); 
train_size=ceil(0.7*sample_size); 
test_size=sample_size-train_size;
rng(my_seed); %Use custom seed in order to replicate randomness 
train_ind=randperm(sample_size,train_size)'; 
test_ind=setdiff(1:sample_size,train_ind)'; 
X_train=bosthous(train_ind,:); %Train features 
X_test=bosthous(test_ind,:); %Test features 
y_train=MEDV(train_ind); %Train values 
y_test=MEDV(test_ind); %Test values
% Now, let center and scale both train and test subsamples
%X_train=X_train(:,[1:3 5:end]);
%X_test=X_test(:,[1:3 5:end]);
X_train_c=zscore(X_train); 
y_train_c=y_train-repmat(mean(y_train),train_size,1); 
X_test_c=(X_test- repmat(mean(X_train),test_size,1))./repmat(std(X_train),test_size,1); 
y_test_c=(y_test-repmat(mean(y_train),test_size,1));
rng(curr_seed); %Set the original seed back

%-------------------------------OLS---------------------------------------- 
%OLS model. MSE for train sample 
B_ols=(inv(X_train_c'*X_train_c))*X_train_c'*y_train_c; 
y_ols=X_train_c*B_ols; 
error_ols_train=sum((y_ols-y_train_c).^2)/train_size;
%MSE for test sample
y_ols=X_test_c*B_ols; 
error_ols_test=sum((y_ols-y_test_c).^2)/test_size;

%------------------------------RIDGE--------------------------------------- 
lambda_ridge=logspace(0,5); %Ridge regularization parameters 
m=length(lambda_ridge);
B_ridge=ridge(y_train_c, X_train_c, lambda_ridge,1); %Constant term is not penalized
%Find error on train sample
y_pred_train=X_train_c*B_ridge; 
y_train_M=repmat(y_train_c,1,m); 
error_train=sum((y_pred_train-y_train_M).^2)/train_size;
%Find error on test sample
y_pred_test=X_test_c*B_ridge; 
y_test_M=repmat(y_test_c,1,m); 
error_test=sum((y_pred_test-y_test_M).^2)/test_size;

%------------------------PLOTS. Ridge MSE---------------------------------- 
%MSE on train sample
figure
semilogx(lambda_ridge,error_train)
hold on
%MSE on test sample 
semilogx(lambda_ridge,error_test,'r-')
hold on
%Optimal constant prediction (mean(y_train_c)) 
error_const_train=sum((repmat(mean(y_train_c),train_size,1)- y_train_c).^2)/train_size; 
semilogx(lambda_ridge,repmat(error_const_train,m,1),'m-') 
error_const_test=sum((repmat(mean(y_train_c),test_size,1)- y_test_c).^2)/test_size; 
semilogx(lambda_ridge,repmat(error_const_test,m,1),'k-') 
title('Ridge regression. MSE');
legend('Ridge train','Ridge test','Const train','Const test'); 
xlabel('Lambda');
ylabel('MSE');

%-----------------------------LASSO---------------------------------
lambda_lasso=logspace(-3,2);
[B_lasso,FitInfo] = lasso(X_train_c,y_train_c,'Lambda',lambda_lasso); 
%Find error on train sample
y_pred_train_l=X_train_c*B_lasso;
lasso_m=length(lambda_lasso);
y_train_M=repmat(y_train_c,1,lasso_m); 
error_train_l=sum((y_pred_train_l-y_train_M).^2)/train_size;
%Find error on test sample
y_pred_test_l=X_test_c*B_lasso;
y_test_M=repmat(y_test_c,1,lasso_m); 
error_test_l=sum((y_pred_test_l-y_test_M).^2)/test_size;

%------------------------PLOTS. LASSO MSE---------------------------------- 
%MSE on train sample
%Plot error lasso
figure
semilogx(lambda_lasso,error_train_l)
%plot(lambda_lasso,FitInfo.MSE)
hold on
%MSE on test sample
semilogx(lambda_lasso,error_test_l,'r')
hold on
%Optimal constant prediction 
semilogx(lambda_lasso,repmat(error_const_train,lasso_m,1),'m-') 
semilogx(lambda_lasso,repmat(error_const_test,lasso_m,1),'k-') 
title('Lasso regression. MSE');
legend('Lasso train','Lasso test','Const train','Const test'); 
xlabel('Lambda');
ylabel('MSE');

%----------------------PLOTS. Ridge Coefficients---------------------------
figure
semilogx(lambda_ridge,B_ridge);
hold on semilogx(lambda_ridge,zeros(m,1),'k'); 
title('Ridge regression. Coefficients') 
legend(varnames);
xlabel('Lambda'); 
ylabel('Coefficients');

%-----------------------PLOTS. Lasso Coefficients--------------------------
figure
semilogx(lambda_lasso,B_lasso);
hold on semilogx(lambda_lasso,zeros(lasso_m,1),'k'); 
title('Lasso regression. Coefficients') 
legend(varnames);
xlabel('Lambda');
ylabel('Coefficients');