clear;
filepath = 'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat';

load(filepath);

writetable(, ('D1.csv'));

% Load the .mat file
matData = load(filepath);

% Extract the variable you want to convert to CSV
% Replace 'yourVariable' with the actual variable name you want to export
yourVariable = matData.yourVariable;

% Specify the path for the output .csv file
csvFilePath = 'path/to/your/output.csv';

% Convert the variable to a CSV file
csvwrite('C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.csv', yourVariable);

disp(['Conversion completed. CSV file saved at: ' csvFilePath]);
