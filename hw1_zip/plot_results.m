clear; close all; clc;
addpath('matlab_func');
addpath('../ccra/results/');
common_settings;
is_printed = true;

log_file = 'log_200_0.0_0.01.csv';
fig_path = '../../ams-559_cse-591_fall2017/hw1/figs/'
figSize = figSizeOneCol;
%% PDF
if true
  [iteration,trainErrors,testErrors,complTimes] = importLogFile(log_file);

  plot(iteration, trainErrors, 'LineWidth', 2);  
  hold on;
  plot(iteration, testErrors, 'LineWidth', 2);
  
  xLabel='iterations';
  yLabel='RMSE';
  
  legend({'train error', 'test error'});

  set (gcf, 'Units', 'Inches', 'Position', figSize, 'PaperUnits', 'inches', 'PaperPosition', figSize);
  xlabel(xLabel,'FontSize',fontAxis);
  ylabel(yLabel,'FontSize',fontAxis);
  xlim([0 max(iteration)]);
  set(gca,'FontSize',fontAxis);

  if is_printed
     figIdx=figIdx +1;
     fileNames{figIdx} = 'error_analysis';
     epsFile = [ LOCAL_FIG fileNames{figIdx} '.eps'];
     print ('-depsc', epsFile);
  end
end

if true
  figure;
  
  log_files = {'log_200_0.0_0.01.csv',
               'log_200_0.05_0.01.csv',
               'log_200_0.1_0.01.csv',               
               'log_200_0.25_0.01.csv',
               'log_200_0.5_0.01.csv',
               'log_200_1_0.01.csv',
               'log_200_2_0.01.csv'
               };
      
  lambda = [0.0 0.05 0.1 0.25 0.5 1]
  minErr = zeros(size(lambda));
  for iFile = 1:length(log_files)
    [iteration,trainErrors,testErrors,complTimes] = importLogFile(log_files{iFile});
    if ~isempty(testErrors)      
      minErr(iFile) = min(testErrors);
    end
  end

%   bar(lambda, minErr, 0.5);  
  plot(lambda, minErr, 'LineWidth', 2);  
  
  xLabel='\lambda';
  yLabel='RMSE';
  
%   legend({ 'test error'});

  set (gcf, 'Units', 'Inches', 'Position', figSize, 'PaperUnits', 'inches', 'PaperPosition', figSize);
  xlabel(xLabel,'FontSize',fontAxis);
  ylabel(yLabel,'FontSize',fontAxis);
  set(gca,'FontSize',fontAxis);

  if is_printed
     figIdx=figIdx +1;
     fileNames{figIdx} = 'reg_analysis';
     epsFile = [ LOCAL_FIG fileNames{figIdx} '.eps'];
     print ('-depsc', epsFile);
  end
end

%%
return;
%% convert to pdf

for i=1:length(fileNames)
    fileName = fileNames{i};
    epsFile = [ LOCAL_FIG fileName '.eps'];
    pdfFile = [ fig_path fileName '.pdf']   
    cmd = sprintf(PS_CMD_FORMAT, epsFile, pdfFile);
    status = system(cmd);
end
