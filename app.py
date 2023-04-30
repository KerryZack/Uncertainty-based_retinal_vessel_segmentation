import cv2
import streamlit as st
from PIL import Image
from clf import predict
import matplotlib.pyplot as plt
import numpy as np
import time
st.set_option('deprecation.showfileUploaderEncoding', False)

# st.title("糖尿病视网膜病变系统")
st.markdown("<h1 style='text-align: center; color: grey;'>糖尿病视网膜病变分割系统</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
# option = st.selectbox(
#      'Choose the model you want to use?',
#      ('resnet50', 'resnet101', 'densenet121','shufflenet_v2_x0_5','mobilenet_v2'))
# ""
# option2 = st.selectbox(
#      'you can select some image',
#      ('image_dog', 'image_snake'))

# st.write("请上传一张图片：")
file_up = st.file_uploader("请上传一张图片")

image = Image.open("image/google.jpg")
st.image(image, caption='Uploaded Image.', use_column_width=True)
if file_up is None:
    # image = Image.open("image/google.jpg")
        # cv2.imread("image/Image_01L.jpg", cv2.IMREAD_COLOR)
    # image = image[:,:,-1]
    # st.write("没有上传图片文件，请上传")
    # if option2 =="image_dog":
    #     image=Image.open("image/dog.jpg")
    #     file_up="image/dog.jpg"
    # else:
    #     image=Image.open("image/snake.jpg")
    #     file_up="image/snake.jpg"
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.warning("上传为空，请上传图像!")
    # st.write()

    fps, prior, std, mask = predict(file_up)
    # labels, fps = predict(file_up, option)

    # print out the top 5 prediction labels with scores
    # st.image(prior, caption='Prior Knowledge', use_column_width=True)

    # for i in labels:
    #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])

    # print(t2-t1)
    # st.write(float(t2-t1))
    st.write("")
    st.metric("", "FPS:   " + str(fps))

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    fps, prior, std, mask = predict(file_up)
    # labels, fps = predict(file_up, option)
    st.success('成功预测，请等待结果显示')

    # print out the top 5 prediction labels with scores
    # st.success('successful prediction')


    # st.pyplot(mask,cmap='gray', vmin=0, vmax=1)

    # st.image(std, caption='Uncertainty Map', use_column_width=True,clamp=True)

    st.write("不确定性结果和分割结果如下：")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(std, cmap=plt.cm.Blues, alpha=0.8)
        fig.colorbar(heatmap, ax=ax)
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        im = ax.imshow(mask, cmap='binary', vmin=0, vmax=1)
        ax.axis('off')  # 隐藏坐标轴
        st.pyplot(fig,use_container_width=True)




    # st.pyplot(fig)


    # st.write("分割结果如下：")
    # fig, ax = plt.subplots()




    # mask =mask*255

    # st.image(mask, caption='Predicts', use_column_width=True, clamp=True)


    # print out the top 5 prediction labels with scores
    # st.success('successful prediction')
    # for i in labels:
    #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])

    # print(t2-t1)
    # st.write(float(t2-t1))
    st.write("")
    st.metric("","FPS:   "+str(fps))
