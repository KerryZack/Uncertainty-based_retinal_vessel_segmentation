# Uncertainty-based_retinal_vessel_segmentation

## 安装步骤如下

conda create -n web python=3.7

activate web

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

pip install streamlit

注意不要使用requirements.txt进行环境安装，因为在部署阶段streamlit cloud也是直接连接到github仓库中，在streamlit cloud中已经存在streamlit的环境。

## 运行代码
在终端切换到当前文件夹的命令下

streamlit run app.py

## 部署
请参看streamlit官方说明，此代码的环境是python3.7，同样需要在streamlit cloud创建中选择好python版本 https://docs.streamlit.io/

## 部署效果
![image](https://github.com/KerryZack/Uncertainty-based_retinal_vessel_segmentation/assets/99378600/aa9772be-24a7-4764-8a06-c6717dbc36fb)

![image](https://github.com/KerryZack/Uncertainty-based_retinal_vessel_segmentation/assets/99378600/13d74b8b-11ad-4998-b69d-688f633ef85e)

## 系统说明
上图中左侧是上传的原始图像，模型推理计算后将显示不确定性图（中间的图）和分割结果（右图）。不确定性图及系统对于分割结果高度不确定性的区域，需要进一步细化诊断。
