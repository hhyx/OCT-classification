t1=clock;
functions = 'test';
filename = {'CNV', 'DME', 'DRUSEN', 'NORMAL'};
path = '../datas/';
file_path = cd;

disp('Renaming Pictures');
for i = 1:4
	cd(file_path)
	path_ = strcat(path, functions, '/', filename(i), '/');
	cd(path_{1})
	files = dir('*.png');
	n = length(files);
	for ifile = 1:n
		oldname = files(ifile).name;
		newname = strcat(filename(i), '-', int2str(ifile-1), '.png');
		file = [strcat('!rename,', oldname, ',', newname)];
		eval(file{1});
	end
end
cd(file_path)
disp('Pictures renamed');

if exist([strcat(path, 'BM3D')], 'dir') == 0
    mkdir([strcat(path, 'BM3D')]);
end

for j = 1:1
    for i = 0:0
        str = strcat(path, functions , '/' , filename(j) , '/' , filename(j) , '-' , int2str(i) , '.png');
        disp(str);
        x = im2double(imread(str{1}));
        [~, x_est] = BM3D(1, x);
%         figure; imshow(x_est);
%         figure; imshow(x);
        str = strcat(path, 'BM3D/' , 'BM3D-' , filename(j) , '-' , int2str(i) , '.png');
        imwrite(x_est, str{1});
    end
end
t2=clock;
etime(t2,t1)