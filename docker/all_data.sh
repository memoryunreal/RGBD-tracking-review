docker run --rm -dit --gpus all -v /data1/yjy/rgbd_benchmark/:/home/dataset/rgbd_benchmark -v /data1/yjy/workspace_votRGBD2019/:/home/dataset/ -v /home/yangjinyu/rgbd_tracker/:/home/rgbd_tracker --name all_rgbd -p 6666:22 watchtowerss/rgbd:latest

