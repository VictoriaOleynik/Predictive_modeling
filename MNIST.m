function [error_test_knn, error_train_avg_knn, error_test_svm, error_train_avg_svm,...
    error_test_dt, error_train_avg_dt, error_test_all_knn, error_train_all_knn,...
    error_test_all_svm, error_train_all_svm,error_test_all_dt,error_train_all_dt] = Classification(my_seed, cv_indices, X_train,X_test, y_train,y_test, varnames)
    train_size=length(y_train);
    test_size=length(y_test);
    y_predict_test_knn=zeros(train_size,1); %Empty vector for predicted values (kNN, test folds in CV)
    y_predict_test_svm=zeros(train_size,1); %Empty vector for predicted values (SVM, test folds in CV)
    y_predict_test_dt=zeros(train_size,1); %Empty vector for predicted values (DT, test folds in CV)
    y_predict_train_knn=zeros(train_size,10); %Empty matrix for predicted values (kNN, train folds in CV)
    y_predict_train_svm=zeros(train_size,10); %Empty matrix for predicted values (SVM, train folds in CV)
    y_predict_train_dt=zeros(train_size,10); %Empty matrix for predicted values (SVM, train folds in CV)
    error_train_knn=zeros(train_size,1);
    error_train_svm=zeros(train_size,1);
    error_train_dt=zeros(train_size,1);
    %--------------------------------kNN--------------------------------------
    %k-parameter calibration
    K=15;%max number of nearest neighbors in kNN
    for k=1:K
        for i=1:10
            cv_test=(cv_indices==i); 
            cv_train=~cv_test; 
            cv_mdl=fitcknn(X_train(cv_train,:),y_train(cv_train,:),'NumNeighbors',k,'Standardize',false);
            y_predict_test_knn(cv_test)=predict(cv_mdl,X_train(cv_test,:));
            y_predict_train_knn(cv_train,i)=predict(cv_mdl,X_train(cv_train,:)); 
            %error in training samples in CV
            for j=1:train_size
                if y_predict_train_knn(j,i)~=0
                    error_train_knn(j)=error_train_knn(j)+(y_predict_train_knn(j,i)~=y_train(j));
                end
            end
        end
        error_test_knn(k)=sum(y_predict_test_knn~=y_train)/train_size;
        error_train_knn=error_train_knn/9;
        error_train_avg_knn(k)=sum(error_train_knn)/train_size;
    end
    figure
    plot(1:K,error_train_avg_knn,'*-');
    hold on
    plot(1:K,error_test_knn,'r*-');
    title('Average errors as function of k in kNN');
    legend('Train samples in CV','Test sample in CV'); xlabel('k parameter in kNN');
    ylabel('Average error');

    %---------------------------------SVM-------------------------------------
    kernel_fun={'rbf','linear','polynomial'};
    for k=1:3
        for i=1:10
            cv_test=(cv_indices==i); 
            cv_train=~cv_test;
            SVMModels = cell(3,1);
            classes = unique(y_train(cv_train,:));
            kernel_now=cell2mat(kernel_fun(k));
            rng(my_seed); % For reproducibility
            for j = 1:numel(classes);
                svm_indx = (y_train(cv_train,:)==classes(j)); % Create binary classes for each classifier
                SVMModels{j} = fitcsvm(X_train(cv_train,:),svm_indx,'ClassNames',[false true],'Standardize',false,...
                'KernelFunction',kernel_now);
            end
            %predicting classes on test folds
            Scores_test = zeros(sum(cv_test),numel(classes));
            for j = 1:numel(classes);
                [~,score] = predict(SVMModels{j},X_train(cv_test,:));
                Scores_test(:,j) = score(:,2); % Second column contains positive-class scores
            end
            [~,maxScore] = max(Scores_test,[],2);
            y_predict_test_svm(cv_test)=maxScore;
            %predicting classes on train folds
            Scores_train = zeros(sum(cv_train),numel(classes));
            for j = 1:numel(classes);
                [~,score] = predict(SVMModels{j},X_train(cv_train,:));
                Scores_train(:,j) = score(:,2); % Second column contains positive-class scores
            end
            [~,maxScore_train] = max(Scores_train,[],2);
            y_predict_train_svm(cv_train,i)=maxScore_train;
            %error in training samples in CV
            for j=1:train_size
                if y_predict_train_svm(j,i)~=0
                    error_train_svm(j)=error_train_svm(j)+(y_predict_train_svm(j,i)~=y_train(j));
                end
            end
        end
        error_test_svm(k)=sum(y_predict_test_svm~=y_train)/train_size;
        error_train_svm=error_train_svm/9;
        error_train_avg_svm(k)=sum(error_train_svm)/train_size;
    end

    %---------------------------Decision trees--------------------------------
    for k=1:5    
        for i=1:10
            cv_test=(cv_indices==i); 
            cv_train=~cv_test; 
            tree=classregtree(X_train(cv_train,:),y_train(cv_train,:),'names',varnames,'method','classification','minleaf',k);
            sfit=eval(tree,X_train(cv_test,:));
            temp_test=zeros(sum(cv_test),1);
            %converting cell array to array of doubles
            for j=1:length(sfit)
                temp_test(j)=str2double(sfit(j));
            end
            y_predict_test_dt(cv_test)=temp_test;
            sfit=eval(tree,X_train(cv_train,:));
            temp_train=zeros(sum(cv_train),1);
            %converting cell array to array of doubles
            for j=1:length(sfit)
                temp_train(j)=str2double(sfit(j));
            end
            y_predict_train_dt(cv_train,i)=temp_train; 
            %error in training samples in CV
            for j=1:train_size
                if y_predict_train_dt(j,i)~=0
                    error_train_dt(j)=error_train_dt(j)+(y_predict_train_dt(j,i)~=y_train(j));
                end
            end
        end
        error_test_dt(k)=sum(y_predict_test_dt~=y_train)/train_size;
        error_train_dt=error_train_dt/9;
        error_train_avg_dt(k)=sum(error_train_dt)/train_size;
    end

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
    rng(my_seed); % For reproducibility
    tree=classregtree(X_train,y_train,'names',varnames,'method','classification','minleaf',1);
    view(tree);
    sfit = eval(tree,X_test);
    error_test_all_dt=sum(sfit~=categorical(y_test))/test_size;
    sfit_train = eval(tree,X_train);
    error_train_all_dt=sum(sfit_train~=categorical(y_train))/train_size;
end

