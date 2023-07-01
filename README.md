# TEZARNet - TEmporal Zero-shot Activity Recognition Network

## Abstract

Most existing Zero-Shot Human Activity Recognition (ZS-HAR) methods based on Inertial Measurement Unit (IMU) data rely on attributes or word embeddings of class labels as auxiliary data to relate the seen and unseen classes. However, defining attributes requires expert knowledge, and both attributes and word embeddings lack motion-specific information. On the other hand, videos depicting various human activities are readily available and contain valuable information for ZS-HAR based on inertial sensor data. This paper proposes a new IMU-data-based ZS-HAR model using videos as auxiliary data. In contrast to the current work, we employ a Bidirectional Long-Short Term based IMU encoder to exploit the temporal information. The proposed model outperforms the state-of-the-art accuracy by 4.7\%, 7.8\%, 3.7\%, and 9.3\% for benchmark datasets PAMAP2, DaLiAc, UTD-MHAD, and MHEALTH, respectively.

## Overview of TEZARNET Training Phase
![tezarnetOverview](https://github.com/nipdep/TEZARNet/blob/main/bilstm_inference.png?raw=true )


