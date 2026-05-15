# RKNN Docker 说明

这个目录是给 Apple Silicon Mac 准备的 RKNN Docker 环境草稿。

思路是:

1. 使用本机 Docker Desktop
2. 使用本地导入的 Ubuntu 22.04 ARM64 rootfs 镜像
3. 在容器里通过 PyPI 安装 `rknn-toolkit2==2.3.2`

基础镜像标签:

- `ubuntu:22.04-arm64-rootfs`

构建命令:

```bash
"/Applications/Docker.app/Contents/Resources/bin/docker" build -t local/rknn-toolkit2:2.3.2 /Users/wuhongfei/workspace/model_install/docker/rknn
```

进入容器:

```bash
"/Applications/Docker.app/Contents/Resources/bin/docker" run --rm -it \
  -v /Users/wuhongfei/workspace/model_install:/workspace \
  local/rknn-toolkit2:2.3.2
```
