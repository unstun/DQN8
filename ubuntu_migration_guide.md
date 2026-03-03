# Ubuntu 系统迁移指南 (rsync 方案)

此文档用于在系统崩溃且能进入 tty1 时，将原系统的根分区安全迁移至 2TB 移动硬盘 (对应设备为 `/dev/sda`)。

## ⚠️ 警告
开始操作前，请确保 `/dev/sda` (移动硬盘) 中没有重要数据！后续的分区和格式化操作将清空该硬盘。

## 第一步：格式化移动硬盘并分区
```bash
# ⚠️ 注意：这会清空 /dev/sda
sudo parted /dev/sda mklabel gpt

# 创建 EFI 分区（512MB）用于引导
sudo parted /dev/sda mkpart primary fat32 1MiB 513MiB
sudo parted /dev/sda set 1 esp on

# 创建根分区（使用剩余所有空间）
sudo parted /dev/sda mkpart primary ext4 513MiB 100%

# 格式化分区
sudo mkfs.fat -F32 /dev/sda1
sudo mkfs.ext4 /dev/sda2
```

## 第二步：挂载目标硬盘
```bash
sudo mkdir -p /mnt/target
# 挂载新根分区
sudo mount /dev/sda2 /mnt/target
# 创建引导挂载点并挂载
sudo mkdir -p /mnt/target/boot/efi
sudo mount /dev/sda1 /mnt/target/boot/efi
```

## 第三步：rsync 拷贝系统数据
通过 rsync 命令将原系统数据完整拷贝至新硬盘（排除虚拟文件系统）：
```bash
sudo rsync -aAXv \
  --exclude={/dev/*,/proc/*,/sys/*,/tmp/*,/run/*,/mnt/*,/lost+found} \
  / /mnt/target/
```
> ⏱ 注：120G 数据大概需要 30–60 分钟，请耐心等待拷贝完成。

## 第四步：配置 GRUB 引导
绑定系统的虚拟文件系统并安装引导程序：
```bash
for d in dev proc sys run; do
  sudo mount --bind /$d /mnt/target/$d
done

# 进入目标系统环境
sudo chroot /mnt/target

# 安装 GRUB 引导器
grub-install --target=x86_64-efi --efi-directory=/boot/efi \
  --bootloader-id=ubuntu --removable /dev/sda

# 更新 GRUB 菜单
update-grub

# 退出 chroot
exit
```

## 第五步：更新 fstab 挂载记录
系统启动需要知道新硬盘的 UUID，需要替换 `/etc/fstab` 中的旧 UUID：
```bash
# 1. 记录下输出的新分区 UUID
sudo blkid /dev/sda1 /dev/sda2

# 2. 编辑 fstab 文件
sudo nano /mnt/target/etc/fstab

# 3. 在 nano 编辑器中：
# 找到旧的 /boot/efi 挂载点，替换为 sda1 的 UUID
# 找到旧的 / 挂载点，替换为 sda2 的 UUID
# 保存并退出 (Ctrl+O, Enter, Ctrl+X)
```

## 第六步：清理并重启测试
解挂所有目录并重启系统：
```bash
# 递归解除挂载
sudo umount -R /mnt/target
sudo reboot
```
> 💡 重启时，请开机前狂按 F2 或 F12 进入 BIOS，选择从 USB 移动硬盘启动。
