% Wavelet denoising
subdir=dir('*.csv');
for i = 1:length(subdir)
    data = readmatrix(subdir(i).name);
    data = fillmissing(data,'previous'); % Fill missing values
    [m,n] = size(data);
    result = zeros(m,n);
    for j = 1:n
        %% Original signal %%
        x=data(:,j);
        subplot(2,1,1);
        plot(x);
        title('Original signal');
        grid on
        %% After wavelet denoising %%
        lev=3;
        xd=wden(x,'minimaxi','s','mln',lev,'haar');
        subplot(2,1,2),
        plot(xd);
        title('Denoised signal');
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