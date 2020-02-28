FROM registry-vpc.cn-hangzhou.aliyuncs.com/eigenlab/yudexcutor:pytorch1.0
COPY requirement.txt /opt/yud/cache/r1.txt
RUN /opt/anaconda/anaconda3/bin/pip install -r /opt/yud/cache/r1.txt
WORKDIR /nfs/users/zhuxuelin/davinci_models
ADD . /nfs/users/zhuxuelin/davinci_models
