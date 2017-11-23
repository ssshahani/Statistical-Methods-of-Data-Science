data = xlsread("final_data.xlsx");
y_GDP = data(:,5);
x1_CPI = data(:,2);
x2_PCE = data(:,3);
x3_PDI = data(:,4);
x4_InterestRate = data(:,1);
x5_Unemployment = data(:,6);

data_predict = xlsread('2017 data.xlsx');
x1_predict_CPI = data_predict(:,2);
x2_predict_PCE = data_predict(:,3);
x3_predict_PDI = data_predict(:,4);
x5_predict_Unemployment = data_predict(:,6);

beta0_predict = ones(3,1);
X_Predict_Final_CPI_PCE_PDI_UE = [beta0_predict x1_predict_CPI x2_predict_PCE x3_predict_PDI x5_predict_Unemployment];
X = [ones(size(y_GDP)) x1_CPI x2_PCE x3_PDI x4_InterestRate x5_Unemployment];
X_PREDICTORS = [x1_CPI, x2_PCE, x3_PDI x4_InterestRate x5_Unemployment];
X_CPI_Regress = [ones(size(y_GDP)) x2_PCE x3_PDI x4_InterestRate x5_Unemployment];
X_PCE_Regress = [ones(size(y_GDP)) x1_CPI	x3_PDI x4_InterestRate x5_Unemployment];
X_PDI_Regress = [ones(size(y_GDP)) x1_CPI	x2_PCE x4_InterestRate x5_Unemployment];
X_Interest_Rate_Regress = [ones(size(y_GDP)) x1_CPI x2_PCE x3_PDI x5_Unemployment];
X_Unemployment_Regress = [ones(size(y_GDP)) x1_CPI x2_PCE x3_PDI x4_InterestRate];
X_Final_CPI_PCE_PDI_UE = [ones(size(y_GDP)) x1_CPI x2_PCE x3_PDI x5_Unemployment];

[beta, y_GDP_hat, epsilon] = performRegression(X,y_GDP);
[sum_RSS, total_variance] = anovaTotal(y_GDP,y_GDP_hat);
correlationPlot(X_PREDICTORS,y_GDP);

VIF(X_CPI_Regress, x1_CPI, "CPI")
VIF(X_PCE_Regress, x2_PCE, "PCE")
VIF(X_PDI_Regress, x3_PDI, "PDI")
VIF(X_Interest_Rate_Regress, x4_InterestRate, "InterestRate")
VIF(X_Unemployment_Regress, x5_Unemployment, "Unemployment")

x_CPI_PCE_PDI = [ones(size(y_GDP)) x1_CPI x2_PCE x3_PDI];
x_PCE_PDI_UE = [ones(size(y_GDP)) x2_PCE x3_PDI x5_Unemployment];
x_CPI_PDI_UE = [ones(size(y_GDP)) x1_CPI x3_PDI x5_Unemployment];
x_CPI_PCE_UE = [ones(size(y_GDP)) x1_CPI x2_PCE x5_Unemployment];
X_PDI_PCE = [ones(size(y_GDP)) x2_PCE x3_PDI]; 
X_PCE_UE = [ones(size(y_GDP)) x2_PCE x5_Unemployment];
X_CPI_PCE = [ones(size(y_GDP)) x1_CPI x2_PCE];
X_PDI_UE = [ones(size(y_GDP)) x3_PDI x5_Unemployment];
X_CPI_PDI = [ones(size(y_GDP)) x1_CPI x3_PDI];
X_CPI_UE = [ones(size(y_GDP)) x1_CPI x5_Unemployment];

Cp_statistic(X_PDI_PCE, y_GDP, epsilon, "PCE_PDI");
Cp_statistic(X_CPI_PCE, y_GDP, epsilon, "CPI_PCE");
Cp_statistic(X_PCE_UE, y_GDP, epsilon, "PCE_UE");
Cp_statistic(X_PDI_UE, y_GDP, epsilon,"PDI_UE");
Cp_statistic(X_CPI_UE, y_GDP, epsilon,"CPI_UE");
Cp_statistic(X_CPI_PDI, y_GDP, epsilon,"PDI_CPI");

Cp_statistic(x1_CPI, y_GDP, epsilon,"CPI");
Cp_statistic(x2_PCE, y_GDP, epsilon, "PCE");
Cp_statistic(x3_PDI, y_GDP, epsilon,"PDI");
Cp_statistic(x5_Unemployment, y_GDP, epsilon,"UE");

Cp_statistic(x_CPI_PCE_UE, y_GDP,epsilon,"CPI_PCE_UE");
Cp_statistic(x_CPI_PDI_UE, y_GDP,epsilon,"CPI_PDI_UE");
Cp_statistic(x_PCE_PDI_UE, y_GDP,epsilon,"PCE_PDI_UE");
Cp_statistic(x_CPI_PCE_PDI,y_GDP,epsilon,"CPI_PCE_PDI");

Cp_statistic(X_Final_CPI_PCE_PDI_UE, y_GDP, epsilon,"CPI_PCE_PDI_UE");

[ HII_Leverage_Vector ] = leveragePoints(X_Final_CPI_PCE_PDI_UE, y_GDP);
studentized_residual_calc(X_Final_CPI_PCE_PDI_UE, y_GDP, HII_Leverage_Vector);
external_studentized_residual(X_Final_CPI_PCE_PDI_UE, y_GDP, HII_Leverage_Vector,epsilon);
CooksDistance(X_Final_CPI_PCE_PDI_UE,y_GDP)
prediction(X_Final_CPI_PCE_PDI_UE, y_GDP, X_Predict_Final_CPI_PCE_PDI_UE);

function [beta, y_GDP_hat, epsilon] = performRegression(X, y_GDP)
    t_X = transpose(X);
    beta = inv(t_X * X) * t_X * y_GDP;

    y_GDP_hat = ones(size(y_GDP));

    for i = 1:size(y_GDP)
        y_GDP_hat(i) =  X(i,:) * beta;
    end
    epsilon = ones(size(y_GDP));

    for i=1:size(y_GDP)
        epsilon(i) = (y_GDP(i) - y_GDP_hat(i));
    end
end

function [sum_RSS, total_variance]  = anovaTotal(y, y_hat)
    mean_y = mean(y);
    sum_RSS = 0;
    for i =1:size(y)
        sum_RSS = sum_RSS + (y_hat(i) - mean_y)^2;
    end
    sum_RSS;

    sum_SSE = 0;
    for i =1:size(y)
        sum_SSE = sum_SSE + (y_hat(i) - y(i))^2;
    end
    sum_SSE;

    total_variance = sum_RSS + sum_SSE;
    total_variance;
end

function correlationPlot(X_PREDICTORS, y_GDP)
    mat = [y_GDP X_PREDICTORS];
    [correlationMatrix, pValue] = corrplot(mat,'varNames',{'GDP', 'CPI', 'PCE','PDI','IR' 'Unemploymed Rate'});
    correlationMatrix;
end

function [Rsquared] = calculateRSquared(y, y_hat)
    [sum_RSS, total_variance] = anovaTotal(y,y_hat);
    Rsquared = sum_RSS/total_variance;
end

function VIF(x_Regress, y, regress)
    [beta, y_hat, epsilon] = performRegression(x_Regress,y);
    Rsquared = calculateRSquared(y, y_hat);
    vif = 1/(1 - Rsquared);
    switch(regress)
        case "CPI"
            vif_CPI = vif;
            display(vif_CPI);
       case "PCE"
            vif_PCE = vif;
            display(vif_PCE);
       case "PDI"
            vif_PDI = vif;
            display(vif_PDI);
       case "InterestRate"
            vif_InterestRate = vif;
            display(vif_InterestRate);
       case "Unemployment"
            vif_Unemployment = vif;
            display(vif_Unemployment);     
    end
end

function Cp_statistic(x, y, epsilon_M, combination)
    [beta, y_GDP_hat, epsilon] = performRegression(x, y);
    sum_SSE = 0;
    for i =1:size(y)
        sum_SSE = sum_SSE + (y_GDP_hat(i) - y(i))^2;
    end
    sum_SSE;
    
    originalmodel_stnd_deviation = std(epsilon_M);
    temp = sum_SSE/originalmodel_stnd_deviation^2;
    
    
    switch(combination)
        case("CPI_PCE_PDI_UE")
            p=4;    
            Cp_statistic_Final = temp - 100 + 2*(p + 1); 
            display(Cp_statistic_Final);
        case("CPI_PCE_UE")
            p=3; 
            Cp_stat_CPI_PCE_Unemployment = temp - 100 + 2*(p + 1);
            display(Cp_stat_CPI_PCE_Unemployment); 
        case("CPI_PDI_UE")
            p=3; 
            Cp_stat_CPI_PDI_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_CPI_PDI_UE); 
        case("PCE_PDI_UE")
            p=3; 
            Cp_stat_PCE_PDI_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_PCE_PDI_UE); 
        case("CPI_PCE_PDI")
            p=3; 
            Cp_stat_CPI_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_CPI_UE);
            
        case("PCE_PDI")
            p=2; 
            Cp_stat_PCE_PDI = temp - 100 + 2*(p + 1);
            display(Cp_stat_PCE_PDI);
        case("CPI_PCE")
            p=2; 
            Cp_stat_CPI_PCE = temp - 100 + 2*(p + 1);
            display(Cp_stat_CPI_PCE);
        case("PCE_UE")
            p=2; 
            Cp_stat_PCE_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_PCE_UE);
        case("PDI_UE")
            p=2; 
            Cp_stat_PDI_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_PDI_UE);
        case("CPI_UE")
            p=2; 
            Cp_stat_CPI_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_CPI_UE);
        case("PDI_CPI")
            p=2; 
            Cp_stat_PDI_CPI = temp - 100 + 2*(p + 1);
            display(Cp_stat_PDI_CPI);
            
        case("CPI")
            p=1; 
            Cp_stat_CPI = temp - 100 + 2*(p + 1);
            display(Cp_stat_CPI);
        case("PDI")
            p=1; 
            Cp_stat_PDI = temp - 100 + 2*(p + 1);
            display(Cp_stat_PDI);
        case("PCE")
            p=1; 
            Cp_stat_PCE = temp - 100 + 2*(p + 1);
            display(Cp_stat_PCE);
        case("UE")
            p=1; 
            Cp_stat_UE = temp - 100 + 2*(p + 1);
            display(Cp_stat_UE);
    end

end

function [Hii] = leveragePoints(x, y)
    t_x_FINAL = transpose(x);
    hat_matrix = x * inv(t_x_FINAL*x) * t_x_FINAL;
    Hii =ones(size(y));
    for i=1:size(y)
        Hii(i) = hat_matrix(i,i);
    end
    figure()
    hold on
    title('Leverage')
    xlabel('Leverage Values')
    ylabel('Frequency')
    dim = [.4 .5 .3 .3];
    str = 'Hii>0.1. Then its High Leverage value.';
    annotation('textbox',dim,'String',str,'FitBoxToText','on');
    histogram(Hii)
    hold off
end

function studentized_residual_calc(x, y, Hii)
    [beta, y_GDP_hat, epsilon] = performRegression(x, y);
    raw_residual = ones(size(y));
    for i=1:size(y)
        raw_residual(i) = (y(i) - y_GDP_hat(i));
    end
    figure
    hold on
    title('Raw Residual')
    xlabel('Raw Residual Values')
    ylabel('Frequency')
    histogram(raw_residual);
    hold off
    
    studentized_residual = ones(size(y));
    s = std(raw_residual);
    for i =1:size(y)
        studentized_residual(i) = raw_residual(i)/s*sqrt(1-Hii(i));
    end
    figure
    hold on
    title('Studentized Residual')
    xlabel('Studentized Residual Values')
    ylabel('Frequency')
    histogram(studentized_residual);
    hold off
end
% q
function external_studentized_residual(x, y, Hii, epsilon_M)
    extnl_studentized_residual = ones(size(y));
    for i =1:size(y)
        epsilon_vector_local = ones(size(y));
        temp_x = x;
        temp_x(i,:) = [];
        temp_y = y;
        temp_y(i,:) = [];
        [beta, y_GDP_hat, epsilon] = performRegression(x, y);
        temp = ones(size(y));
        sum =0;
        n=100;
        p=4;
        for j=1:size(y)-1
            epsilon_vector_local(j) = temp_y(j) - y_GDP_hat(j);
        end
        for j=1:size(y)-1
            sum = sum + (epsilon_vector_local(j) - mean(epsilon_vector_local))^2;
        end
        var = sum/(n-p+1);
        extnl_studentized_residual(i)=epsilon_M(i)/sqrt(var*(1-Hii(i)));
    end
    figure
    hold on 
    title('RStudent (External Studentized Residual)')
    xlabel('Rtudent (External Studentized Residual) Values')
    ylabel('Frequency')
    histogram(extnl_studentized_residual);
    hold off
end

function CooksDistance(x, y)
    epsilon_vector_local = ones(size(y));
    [beta, y_all_hat, epsilon] = performRegression(x, y);
    sum = 0;
    cooksD = ones(size(y));
    p=4;
    for i =1:size(y)-1
        temp_x = x;
        temp_x(i,:) = [];
        temp_y = y;
        temp_y(i,:) = [];
        [beta, y_less_1_hat, epsilon] = performRegression(temp_x, temp_y);
        for j=1:size(y)-1
           epsilon_vector_local(j) = temp_y(j) - y_less_1_hat(j);
        end
        s = std(epsilon_vector_local);
        sum =sum + (y_all_hat(i) - y_less_1_hat(i))^2;
        cooksD(i) = sum/((p+1)*s^2);
    end
    figure
    hold on 
    title('Cook-s Distance')
    xlabel('Cooks Distance Values')
    ylabel('Frequency')
    histogram(cooksD)
    hold off    
end

function prediction(X_Final, y_GDP, X_Predict_Final_CPI_PCE_PDI_UE)
    t_X_Final = transpose(X_Final);
    beta = inv(t_X_Final * X_Final) * t_X_Final * y_GDP;

    y_GDP_hat_2017 = ones(3,1);

    for i = 1:(3)
        y_GDP_hat_2017(i) =  X_Predict_Final_CPI_PCE_PDI_UE(i,:) * beta;
    end
    epsilon = ones(3);
    
    display(y_GDP_hat_2017);

    for i=1:(3)
        epsilon(i) = (y_GDP(i) - y_GDP_hat_2017(i));
    end
end