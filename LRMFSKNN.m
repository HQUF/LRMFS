
base_path = '**';
save_path = '';
file_name = {****}; 
                    
file_format = '.mat';


for jj =31%[1:5,7:13,19:32]
    jj
    
    file_name{jj}
    
    set = [file_name{jj},file_format];
    finaldata = [];
    Num=10;
    Smooth=1;                                                                                                                                                                                                                   
    %% 导入数据
    path = [base_path,set];
    dataset = load(path);
    if(isfield(dataset,'train_data'))
        temp_data = [dataset.train_data; dataset.test_data];
        temp_target = [dataset.train_target, dataset.test_target];
    else
        temp_data = zscore(dataset.data);
        temp_target = dataset.target;
    end
    temp_target(temp_target==-1)=0;
    
    options.alpha = 0.1;
    options.beta = 1;
    options.gamma= 1;
    options.lambda= 1;
    
        [rank] = LRMFS(temp_data',temp_target',options);
    
    
    for t=1:20
    t
    ratio = 0.01;
    data = temp_data(:,rank(1:floor(size(temp_data,2)*ratio*t)));
    target = temp_target;
    %data = zscore(data);
    target(target==0)=-1;
    [data_num,tmp] = size(data);
    for i = data_num:-1:1
        swap_num = randi([1,i],1,1);
        tmpd = data(i,:);
        tmpt = target(:,i);
        data(i,:) = data(swap_num,:);
        target(:,i) = target(:,swap_num);
        data(swap_num,:) = tmpd;
        target(:,swap_num) = tmpt;
    end
    %交叉验证次数
    fold_num = 5;
    test_num = round(data_num/fold_num);
    test_instance = cell(fold_num,1);
    for i = 1:fold_num-1
        test_instance{i,1} = (i-1)*test_num+1:i*test_num;
    end
    test_instance{fold_num,1} = (fold_num-1)*test_num+1:data_num;

%     ratio = 0.1;
%     mu = 0.1;
%     projtype = 'proj';
    result = zeros(fold_num,7);
    for i = 1:fold_num
        disp(['The ',num2str(i),'-th fold is going on...']);
        train_data = data;
        train_target = target;
        test_data = data(test_instance{i,1},:);
        test_target = target(:,test_instance{i,1});
        train_data(test_instance{i,1},:) = [];
        train_target(:,test_instance{i,1}) = [];
        
        [Prior,PriorN,Cond,CondN] = MLKNN_train(train_data,train_target,Num,Smooth); 
        [Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
        
        
        HammingLoss=Hamming_loss(Pre_Labels,test_target);
        RankingLoss=Ranking_loss(Outputs,test_target);
        OneError=One_error(Outputs,test_target);
        Coverage=coverage(Outputs,test_target);
        Average_Precision=Average_precision(Outputs,test_target);
        Macro_f1=MacroF1(test_target,Pre_Labels);
        Micro_f1=MicroF1(test_target,Pre_Labels);
        result(i,:) = [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Macro_f1,Micro_f1];
    end

        H_Mean(jj,t) = mean(result(:,1));
        H_Std(jj,t) = std(result(:,1));
        R_Mean(jj,t) = mean(result(:,2));
        R_Std(jj,t) = std(result(:,2));
        O_Mean(jj,t) = mean(result(:,3));
        O_Std(jj,t) = std(result(:,3));
        C_Mean(jj,t) = mean(result(:,4));
        C_Std(jj,t) = std(result(:,4));
        A_Mean(jj,t) = mean(result(:,5));
        A_Std(jj,t) = std(result(:,5));
        Ma_Mean(jj,t) = mean(result(:,6));
        Ma_Std(jj,t) = std(result(:,6));
        Mi_Mean(jj,t) = mean(result(:,7));
        Mi_Std(jj,t) = std(result(:,7));
    end
end


