FROM chainer/chainer:v3.0.0
LABEL maintainer "rootpia"

RUN apt update &&\
    apt install -y libjpeg-dev libpng-dev python-opencv &&\
    pip install flask flask-cors Pillow

COPY vgg16 /tmp/vgg16
WORKDIR /tmp/vgg16
#RUN python train_mnist.py -u 100 -e 5

EXPOSE 5000

# entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
