# OCT-classification

---

## 运行环境
- MATLAB R2017a
- Python 3.7
- Pytorch 1.1.0
- [原数据集](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- [处理后数据集](https://pan.baidu.com/s/1czj9-FOkGriljx8I0p4k8A)    提取码：psgi

## 代码结构
- BM3D  
  - `BM3D.m`
  	- MATLAB版本BM3D去燥程序
  - `BM3D_progress.m` 
  	- 该代码加载数据集，选择处理的图片路径和数量
- ImageProcess
  - `ImagePreprocess.py`
  	- 图像预处理程序
  - `BM3D.py`
  	- Python版本BM3D去燥程序
  - `Binaryzation.py`
  	- 图像填充，阈值过滤
  - `MedianFilter.py`
  	- 中值滤波，保留图像最大连通域
  - `MorphologicalOpening.py`
  	- 形态学开运算
  - `MorphologicalClosing.py`
  	- 形态学闭运算
  - `Fitting.py`
  	- 线性拟合和二阶多项式拟合
  - `Normalization.py`
  	- 归一化，裁剪
- FeatureExtraction
	- `SIFT.py`
		- 使用SIFT和K-Means方法进行特征提取，训练支持向量机、随机森林进行图像分类
	- `Predict.py`
		- 支持向量机、随机森林进行图像分类预测
- ResNet
  + `base_dataset.py`
    	+ 该python代码是为了加载数据集。将图片载入训练预测试。         
  + `test.py`      
    	+ 测试集相关代码
  + `train.py`
    	+ 训练训练集相关代码
  + `utils.py`
    	+ 显示运行结果的画图的相关函数

## 代码运行
- BM3D  
  - 修改 `BM3D_progress.m` 文件中需要处理的文件目录和文件数量
  - 命令行运行： `matlab -nosplash -nodesktop -r BM3D_progress `
  - 或启动MATLAB，运行 `BM3D_progress.m`

- ImageProcess
  - 修改`ImagePreprocess.py`文件中需要处理的文件目录和文件数量
  - 运行：`python ImagePreprocess.py` 

- FeatureExtraction
  - 使用SIFT和K-Means方法进行特征提取，导出字典（训练集不变时可以导入历史字典进行训练），训练支持向量机和随机森林，导出模型
  - 运行：`python SIFT.py` 
  - 导入字典和模型进行测试
  - 运行：`python Predict.py` 

- PCANet

  - 修改图片路径 `OCT_classification.m`
  - 运行： `OCT_classification.m`

- ResNet
  - 训练

    运行：`python train.py` 

  - 测试

    运行：`python test.py`

  - ResNet数据集说明

    + 为区分图片标签，程序需要从文件夹名来确定图片类型，所有数据集的结构类型应该为
    + ```bash
      ├─datas
      │  ├─train_data	
      │  │ ├─CNV
      │  │ ├─DME
      │  │ ├─DRUSEN
      │  │ ├─NORMAL
      │  ├─test_data		
      │  │ ├─CNV
      │  │ ├─DME
      │  │ ├─DRUSEN
      │  │ ├─NORMAL
      │  ├─val_data	
      │  │ ├─CNV
      │  │ ├─DME
      │  │ ├─DRUSEN
      │  │ ├─NORMAL
      ```

    + 代码运行时，需要将test.py和train.py的`data_dir = '../datas/'`设置为你自己存储的训练集和测试集的数据集路径，val_data文件夹中需要存出一些图片来作为每一次训练之后的测试图片，以此来估算分析训练的效果，可以任意设置。

---

## 目录结构

```bash

├─OCT              	 	// OCT图像分类程序 
│  ├─BM3D				// BM3D去燥程序         
│  ├─ImageProcess		// 图像预处理    
│  ├─FeatureExtraction	// 使用SIFT和K-Means方法进行特征提取，支持向量机、随机森林进行图像分类
│  ├─PCANet				// PCANet特征提取,支持向量机进行图像分类
│  └─ResNet				// 残差神经网络（ResNet34）对图像进行分类
├─README.md				// README
├─实训报告               // 小组实训报告 
├─期末总结	         	 // 期末总结PPT
```