
% The files are the MATLAB source code for the paper:
% Feng Gao,  Junyu Dong, Bo Li, Qizhi Xu, Cui Xie. 
% Change Detection from Synthetic Aperture Radar Images Based on
% Neighborhood-Based Ratio and Extreme Learning Machine 
% Journal of Applied Remote Sensing. 10(4), 2016.
%
% The demo has not been well organized. 
% Please contact me if you meet any problems.
% 
% Email: gaofeng@ouc.edu.cn
% 
% ע�����
%   ��Ϊѵ��ʱ������������ɣ�������ͬ��Ӱ�����յĽ����
%   ����ÿ�����еĽ��������һ����
%   ���߻��ں��ڷ����İ汾�н����������̶���������������ӣ�
%



clear;
clc;
close all;

addpath('./Utils');
addpath('NBR');
addpath SPBL;
% PatSize ����Ϊ����
PatSize = 5;
k_n = 3;

fprintf(' ... ... read image file ... ... ... ....\n');
im1   = imread('./pic/ice_part2_1.bmp');
im2   = imread('./pic/ice_part2_2.bmp');
im_gt = imread('./pic/ice_part2_gt.bmp');
fprintf(' ... ... read image file finished !!! !!!\n\n');

im1 = double(im1(:,:,1));%ȡRGBֵ 1��ʾ��ɫ �ڻҶ�ͼ��rgb����һ����
im2 = double(im2(:,:,1));
im_gt = double(im_gt(:,:,1));

[ylen, xlen] = size(im1);

% �� neighborhood-based ratio image
fprintf(' ... .. compute the neighborhood ratio ..\n');
nrmap = nr(im1, im2, k_n);
nrmap = max(nrmap(:))-nrmap;
nrmap = nr_enhance( nrmap );
feat_vec = reshape(nrmap, ylen*xlen, 1);
fprintf(' ... .. compute finished !!! !!! !!! !!!!\n\n');


fprintf(' ... .. clustering for sample selection begin ... ....\n');
im_lab = gao_clustering(feat_vec, ylen, xlen);
fprintf(' ... .. clustering for sample selection finished !!!!!\n\n');

fprintf(' ... ... ... samples initializaton begin ... ... .....\n');
fprintf(' ... ... ... Patch Size : %d pixels ... ....\n', PatSize);

% ��ȡ lab ��Ϣ
pos_lab = find(im_lab == 1);
neg_lab = find(im_lab == 0);

% ��������������˳��
pos_lab = pos_lab(randperm(numel(pos_lab)));
neg_lab = neg_lab(randperm(numel(neg_lab)));

[ylen, xlen] = size(im1);

% ͼ����Χ���㣬Ȼ��ÿ��������ΧȡPatch������
mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im1; 
im1 = im2col_general(imTmp, [PatSize, PatSize]);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im2; 
im2 = im2col_general(imTmp, [PatSize, PatSize]);
%clear imTmp mag;

% �ϲ������� im
im1 = mat2imgcell(im1, PatSize, PatSize, 'gray');
im2 = mat2imgcell(im2, PatSize, PatSize, 'gray');
parfor idx = 1 : numel(im1)
    im_tmp = [im1{idx}; im2{idx}];
    im(idx, :) = im_tmp(:);
end
%clear im1 im2 idx;

% ���ѡ�����أ����ձ���ɸѡ��������
fprintf(' ... ... ... randomly generation samples ... ... .....\n');
PosNum = numel(pos_lab);
NegNum = numel(neg_lab)*0.5;


% ȡ����������ͼ���
PosPat = im(pos_lab(1:PosNum), :);
NegPat = im(neg_lab(1:NegNum), :);
trainfea = [PosPat; NegPat];
traingnd = [ones(PosNum, 1); (zeros(NegNum, 1))+2];%����Ϊ1 ����Ϊ0
trn_data = [traingnd, trainfea];
clear PosPat NegPat  ; 
%clear PosNum NegNum;
clear pos_lab neg_lab;

%TstLab = ones(size(im,1), 1);%traingnd = [ones(size(0:im/2), 1);zeros(im/2+1:end),1];
%tst_data = [TstLab, im];
fea=trainfea;gnd=traingnd;
save('fea','fea','gnd');
% Set the training and validation sample proportion
train_ratio = 0.6;
vali_ratio = 0.0;
n_ratio = 0;
 maxBase = 20;
fprintf(' ============== SPBL begin ========\n');

Gen_Split;
SPBLtrain;
fprintf(' ============== SPBL finished !!!!!\n\n');
idx = find(im_lab == 0.5);
for i = 1:numel(nClass)
    if nClass(i) == 1;
        im_lab(idx(i)) = 1;
    else
        im_lab(idx(i)) = 0;
    end
end
[im_lab,num] = bwlabel(~im_lab);
for i = 1:num
    idx = find(im_lab==i);
   if numel(idx) <= 10
      im_lab(idx)=0;
   end
end
im_lab = im_lab>0;


clear i num trainfea traingnd NumSam;
[FA,MA,OE,CA] = DAcom(im_gt, im_lab);

% ������
fid = fopen('rec.txt', 'a');
fprintf(fid, 'PatSize = %d\n', PatSize);
fprintf(fid, '�龯����: %d \n', FA);
fprintf(fid, '©������: %d \n', MA);
fprintf(fid, '�������: %d \n', OE);
fprintf(fid, '׼ȷ��:   %f \n\n\n', CA);
fclose(fid);

fprintf(' ===== Written change detection results to Rec.txt ====\n\n');









