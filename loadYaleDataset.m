%% Copyright (c) <2016> Pei Xu xuxx0884@umn.edu
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% Author: Pei Xu xuxx0884@umn.edu
% Version: 1.0 10/22/2016

function [train_mat, test_mat] = loadYaleDataset(img_height, img_width)
%% This function read the data of all 38 subjects in CroppedYale database.
%
%   For each subject, randomly pick 12 images are stored as the training data,
% while the rest 3 image are stored as the traingin data. Each image has the
% size 192x168 pixel and will be resized according to img_height, img_width.
%
%   train_mat is a img_height*img_height-by-456 matrix. Each column comes
% from one image. Every 12 columns come from one subject.
%   test_mat is a img_height*img_height-by-114 matrix. Each column comes
% from one image. Every 3 columns come from one subject.
%

data_dir_name = 'data';
data_dirs = dir('data');
dimension = img_height*img_width;
train_mat = zeros(dimension, 456);
test_mat = zeros(dimension, 114);
tr_indx = 0;
test_indx = 0;

s = RandStream('mt19937ar','Seed',0);

for i = 1:numel(data_dirs)
    dir_name = data_dirs(i).name;
    if (length(dir_name) < 5 || ~strcmp('yaleB', dir_name(1:5)))
        continue;
    end
    img_dir = fullfile(data_dir_name, dir_name);
    img_list = dir(fullfile(img_dir, '*.png'));
    randnum = randperm(s, 15, 3);
    for j = 1:15
        img_name = img_list(j).name;
        im = imresize(imread(fullfile(img_dir, img_name)), [img_height, img_width]);
        img = reshape(double(im), dimension, 1);
        if (find(randnum == j))
         test_indx = test_indx + 1;
         test_mat(:, test_indx) = img;
        else
         tr_indx = tr_indx + 1;
         train_mat(:, tr_indx) = img;
        end
    end
end

end

