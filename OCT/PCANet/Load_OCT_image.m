function [ Datas ,Labels] = Load_OCT_image( Directory, Num )
%Load_OCT_image Load_OCT_image
%   此处显示详细说明

Type = ["CNV", "DME", "DRUSEN", "NORMAL"];
TypeNum = 4;
ImageSize = 100;
Datas = [];
Labels = [];
for i = 1: TypeNum
    for j = 0: Num - 1
        ImageName = Directory  + Type(i) + '/Result-' + Type(i) + '-' + num2str(j) + '.png';
        ImageName = char(ImageName);
        Image = imread(ImageName);
%         ImageGray = rgb2gray(Image);
%         ImageGray = imresize(ImageGray,[Height Width], 'bilinear');
        Image = imresize(Image,[ImageSize ImageSize], 'bilinear');
        ImageData = reshape(Image, ImageSize * ImageSize, 1);
        ImageData = double(ImageData ./ 255.0);
        Datas = [Datas, ImageData];
        Labels = [Labels; i];
    end
end


end

