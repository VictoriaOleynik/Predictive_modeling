%---------------------------SAMPLE PREPARATION------------------------
wine=csvread('wine.csv');
%1st attribute in the original sample is class identifier (1-3)
y=wine(:,1);
X=wine(:,2:end);
varnames={'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',...
'Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315','Proline'};
curr_seed=rng; %Save current seed 
my_seed=13; %Set custom seed 
sample_size=length(X(:,1)); 
train_size=ceil(0.7*sample_size); 
test_size=sample_size-train_size; 
rng(my_seed); %Use custom seed in order to replicate randomness 
train_ind=randperm(sample_size,train_size)'; 
test_ind=setdiff(1:sample_size,train_ind)'; 
X_train=X(train_ind,:); %Train features 
X_test=X(test_ind,:); %Test features 
y_train=y(train_ind); %Train values 
y_test=y(test_ind); %Test values 
%-----------------------------10-fold CV----------------------------------
rng(my_seed); %for reproducibility of crossvalind function
cv_indices = crossvalind('Kfold',train_size,10);

%--------------------------------kNN--------------------------------------
%k-parameter calibration
K=15;%max number of nearest neighbors in kNN
[error_train_avg, error_test_knn]=knn(cv_indices,X_train,y_train,K,my_seed,train_size);
figure
plot(1:K,error_train_avg,'*-');
hold on
plot(1:K,error_test_knn,'r*-');
title('Average errors as function of k in kNN');
legend('Train samples in CV','Test sample in CV'); xlabel('k parameter in kNN');
ylabel('Average error');

%---------------------------------SVM-------------------------------------
[error_test_svm, error_train_avg_svm] = svm(cv_indices, X_train, y_train, train_size, my_seed);

%---------------------------Decision trees--------------------------------
[error_test_dt, error_train_avg_dt] = dt(X_train, y_train, cv_indices, my_seed, train_size, varnames);


%------------------------------Whole sample-------------------------------
%--------------------------------kNN, k=1---------------------------------
cv_mdl=fitcknn(X_train,y_train,'NumNeighbors',1,'Standardize',false);
y_predict_all_test_knn=predict(cv_mdl,X_test);
y_predict_all_train_knn=predict(cv_mdl,X_train);
error_test_all_knn=sum(y_predict_all_test_knn~=y_test)/test_size;
error_train_all_knn=sum(y_predict_all_train_knn~=y_train)/train_size;

%----------------------------------SVM------------------------------------
SVMModels = cell(3,1);
classes = unique(y_train);
rng(my_seed); % For reproducibility
for j = 1:numel(classes);
    svm_indx = (y_train==classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X_train,svm_indx,'ClassNames',[false true],'Standardize',false,...
    'KernelFunction','linear');
end
%predicting test classes
Scores = zeros(test_size,numel(classes));
for j = 1:numel(classes);
   [~,score_all_test] = predict(SVMModels{j},X_test);
   Scores(:,j) = score_all_test(:,2); % Second column contains positive-class scores
end
[~,maxScore_all_test] = max(Scores,[],2);
error_test_all_svm=sum(maxScore_all_test~=y_test)/test_size;
%predicting training classes        
Scores = zeros(train_size,numel(classes));
for j = 1:numel(classes);
   [~,score_all_train] = predict(SVMModels{j},X_train);
   Scores(:,j) = score_all_train(:,2); % Second column contains positive-class scores
end
[~,maxScore_all_train] = max(Scores,[],2);
error_train_all_svm=sum(maxScore_all_train~=y_train)/train_size;

%-------------------------------Decision trees-----------------------------
tree=classregtree(X_train,y_train,'names',varnames,'method','classification','minleaf',1);
view(tree);
sfit = eval(tree,X_test);
error_test_all_dt=sum(sfit~=categorical(y_test))/test_size;
sfit_train = eval(tree,X_train);
error_train_all_dt=sum(sfit_train~=categorical(y_train))/train_size;


