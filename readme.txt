conda create -n stn_track python=3.8 -y
conda activate stn_track
conda install pytorch=1.7.1 torchvision cudatoolkit -y               conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
python setup.py develop
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
训练
python tools/train.py -f exps/example/mot/stn_yolox_s_mix_det.py -d 1 -b 8 -c pretrained/yolox_s.pth 
验证
python tools/track.py -f exps/example/mot/stn_yolox_s_mix_det.py -c pretrained/uavbest_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse --match_thresh 0.7
演示
python tools/demo_track.py video -f exps/example/mot/stn_yolox_s_mix_det.py -c pretrained/uavbest_ckpt.pth.tar --path ./videos/M1302.avi --fp16 --fuse --save_result