%小波去噪
subdir=dir('*.csv');
for i = 1:length(subdir)
    data = readmatrix(subdir(i).name);
    data = fillmissing(data,'previous'); %填补空缺值
    [m,n] = size(data);
    result = zeros(m,n);
    for j = 1:n
        %% 原始信号 %%
        x=data(:,j);
                subplot(2,1,1);
                plot(x);
                title('原始信号');
                grid on
        %% 小波消噪后 %%
        lev=3;
        xd=wden(x,'minimaxi','s','mln',lev,'haar');%db2 haar sym2 coif3 bior1.1
                subplot(2,1,2),
                plot(xd);
                title('小波去噪后的信号')
                grid on

        result(:,j)=xd; 
    end 
    folder = './te_haar3/';
    if ~exist(folder,'dir')
        mkdir(folder)
    end
    str=strcat('te_haar3/filter(haar3)3_',subdir(i).name);
    writematrix(result,str);
end

