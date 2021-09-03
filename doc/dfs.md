### 分布式文件系统

DFS允许通过网络访问多主机sharing的文件。

[介绍](https://www.weka.io/learn/distributed-file-system/)

[这里](https://en.wikipedia.org/wiki/Comparison_of_distributed_file_systems)是一些比较，还有[选型](https://bbs.huaweicloud.com/blogs/detail/243039)。



### [glusterfs](https://www.gluster.org/)

#### [安装](https://docs.gluster.org/en/latest/Quick-Start-Guide/Quickstart/)

要点：

1. 在每台主机上，安装

    `apt install glusterfs-server`

2. 在每台主机上，格式化用来做存储的设备，挂载brick，如

   `sbin/mkfs.ext4 /dev/xvdb && mkdir -p /data/brick && mount /dev/xvdb /data/brick`

3. 在其中一台主机上，探测其它每台主机：

   ```bash
   for host in "${other_hosts}"
     sudo gluster peer probe $host
   
   gluster peer status  # 查看状态
   ```

4. 在任一台主机上创建卷：

   ```bash
   create_volume_cmd="gluster volume create my-vol "
   for host in "${all_host}"
   do
     create_volume_cmd+="$host:/data/brick "  # 把所有的bricks联合起来
   done
   eval $create_volume_cmd  # 执行命令
   gluster volume start my-vol  # 启动刚创建的卷
   gluster volume info  # 查看卷信息
   ```

5. 在客户机(可以不是上面主机中的任何一台)上，挂载到的本地某个目录

   ```bash
   mkdir -p /mnt/my_path
   mount -t glusterfs host1:/my-vol /mnt/my_path
   
   # 至此，就可以通过/mnt/my_path来做各种操作了
   ```

