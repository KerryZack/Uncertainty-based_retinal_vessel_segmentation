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


