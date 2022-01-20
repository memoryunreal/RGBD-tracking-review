docker run --rm -dit --gpus all -v /data1/yjy/rgbd_benchmark/:/home/dataset/rgbd_benchmark -v /data1/yjy/workspace_votRGBD2019/:/home/dataset/ -v /home/yangjinyu/rgbd_tracker/LTDSEd/LTDSEd:/home/LTDSEd -v /home/yangjinyu/rgbd_tracker/SiamDW_D:/home/SiamDW_D -v /home/yangjinyu/rgbd_tracker/siammds_submit/:/home/siammds_submit --name rgbdtracker -p 6666:22 watchtowerss/rgbd:latest

