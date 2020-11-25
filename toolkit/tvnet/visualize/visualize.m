flow_file = '../result/result-pytorch.mat';
load(flow_file);
img = flowToColor(flow);
figure;
imshow(img)
% imwrite(img,'result-pytorch.png')

