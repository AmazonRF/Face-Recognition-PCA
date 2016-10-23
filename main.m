% This file uses reduced Cropped Yale face database to do a test of face
% recognition. PCA is used to reduce dimension. Euclidean distance is used
% in recognition.
%
% This file will automatically generate some figures in jpeg format as the
% test result. Close it if you would not like.

%% Global Parameters

% Show eigenfaces
show_eigfaces = true;
% show mean_face
show_meanface = true;
% show eigenvalue distribution
show_eigval = true;
% Show recognition result into figures
show_fig_result = true;

% This is the original size of the image.
% It may be too big for some computers when SVD.
img_height = 168;
img_width = 192;
dimension = img_height * img_width;

% Every 3 columns in test_data belong to one subject
num_e_test = 3;
% Every 12 columns in training_data belong to one subject
num_e_train = 12;

% The dir to store the test results
result_dir = fullfile('test');

%% Load Data
[train_data, test_data] = loadYaleDataset(img_height, img_width);

%% Mean Face
mean_face = mean(train_data, 2);
if (show_meanface == true)
    figure(1);
    imagesc(reshape(mean_face, img_height, img_width));
    colormap('gray');
    title('''Mean'' Face');
end

%% PCA
Xtr = bsxfun(@minus, train_data, mean_face);
[eigvec, r_eigval, ~] = svd(Xtr./sqrt(size(Xtr,2)-1));
d_eigval = diag(r_eigval).^2;
% Normalize eigenvalues
eigval = d_eigval./sum(d_eigval);
cum_eigval = cumsum(eigval);

% Choose the first eig_indx of principal components
indx = find(cum_eigval>0.99);
eig_indx = indx(1);

if (show_eigval == true)
    figure(2);
    subplot(2, 1, 1);
    plot(eigval);
    title('Eigenvalues obtained when PCA');
    xlim([0, length(eigval)]);
    subplot(2, 1, 2);
    plot(cum_eigval);
    title('Cumulative Summation of Eigenvalues obtained when PCA');
    xlim([0, length(eigval)]);
end

fprintf('The first %d principal components are determined to be chosen.\n', eig_indx);

clear r_eigval d_eigval indx;

%% Eigenfaces
eigfaces = zeros(dimension, eig_indx);

% Normalize eigenvectors into eigfaces
for i = 1:eig_indx
    eigfaces(:, i) = eigvec(:, i) ./ sum(eigvec(:, i));
end

% Show first 9 eigenfaces
if (show_eigfaces == true)
    figure(3);
    for i=1:9
        subplot(3, 3, i);
        imagesc(reshape(eigfaces(:, i), img_height, img_width));
        colormap('gray');
        title(sprintf('Eigenface %d', i));
    end
end

%% Transform Training Data
Tr = eigfaces'*Xtr;

%% Transform Test Data
Xte = bsxfun(@minus, test_data, mean_face);
Te = eigfaces'*Xte;

%% Process Matching
% Use Euclidean distance
dist = bsxfun(@plus, -2*Te'*Tr, sum(Tr.*Tr));
[match_rate, indx] = min(dist, [], 2);

%% Generate Result

% Reconstruct Image
re_Tr = eigfaces * Tr;
re_Te = eigfaces * Te;

matched = 0;
unmatched = 0;
unmatched_list = zeros(length(indx), 1);
for i=1:length(indx)
    subj = ceil(i/num_e_test);
    if (subj == ceil(indx(i)/num_e_train))
        matched = matched+1;
        res ='Matched';
    else
        unmatched = unmatched + 1;
        unmatched_list(unmatched) = i;
        res ='Unmatched';
    end
    if (show_fig_result == true)
        fig = figure('Visible', 'off');

        subplot(2, 2, 1);
        imagesc(reshape(test_data(:,i), img_height, img_width));
        colormap('gray');
        title('Test Image');

        subplot(2, 2, 3);
        imagesc(reshape(bsxfun(@plus, re_Te(:,i), mean_face), img_height, img_width));
        colormap('gray');
        title('Reconstructed Test Image');

        subplot(2, 2, 2);
        imagesc(reshape(train_data(:, indx(i)), img_height, img_width));
        colormap('gray');
        title('Matched Training Image');

        subplot(2, 2, 4);
        imagesc(reshape(bsxfun(@plus, re_Tr(:,indx(i)), mean_face), img_height, img_width));
        colormap('gray');
        title('Reconstructed the Matched Training Image');

        suptitle(sprintf('Test Object %d\n%s', i, res));
        % suptitle could force visible set to on
        set(fig, 'Visible', 'off');
        saveas(fig, fullfile(result_dir, sprintf('test_obj_%d', i)), 'jpg');
    end
end
clear i fig subj res;

fprintf('Test Obj: %d\n', length(indx));
fprintf('Matched: %d\n', matched);
fprintf('Accuracy: %f\n', matched/length(indx));
unmatched_list = unmatched_list(1:unmatched);
fprintf('Unmatched:\n')
unmatched_list'

