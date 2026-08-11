#!/bin/bash
# install_python_gdb.sh

set -e

echo "=== 安装 python-gdb 扩展 ==="

# 检测操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo "无法检测操作系统"
    exit 1
fi

echo "操作系统: $OS $VERSION"

install_ubuntu() {
    echo "Ubuntu/Debian 系统安装..."
    apt-get update
    
    # 安装基本工具
    apt-get install -y gdb git wget
    
    # 安装 Python 调试包
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "检测到 Python 版本: $PYTHON_VERSION"
    
    # 尝试安装对应版本的包
    if apt-cache show "python${PYTHON_VERSION}-dbg" > /dev/null 2>&1; then
        apt-get install -y "python${PYTHON_VERSION}-dbg"
    else
        apt-get install -y python3-dbg
    fi
    
    # 尝试安装 python-gdb 包
    if apt-cache show "python${PYTHON_VERSION}-gdb" > /dev/null 2>&1; then
        apt-get install -y "python${PYTHON_VERSION}-gdb"
    else
        echo "未找到 python-gdb 包，将从源码获取"
    fi
    
    # 安装 glibc 调试符号
    apt-get install -y libc6-dbg
}

install_centos() {
    echo "RHEL/CentOS 系统安装..."
    
    # 启用 debuginfo 仓库
    if [ -f /etc/redhat-release ]; then
        if grep -q "CentOS Linux release 7" /etc/redhat-release; then
            yum install -y yum-utils
            yum-config-manager --enable debug
        elif grep -q "CentOS Linux release 8" /etc/redhat-release; then
            dnf install -y dnf-utils
            dnf config-manager --set-enabled debuginfo
        fi
    fi
    
    # 安装包
    yum install -y gdb git wget
    
    # 安装 debuginfo
    if command -v debuginfo-install >/dev/null 2>&1; then
        debuginfo-install -y python3
    else
        yum install -y python3-debuginfo
    fi
    
    yum install -y glibc-debuginfo
}

install_from_source() {
    echo "从源码安装 python-gdb.py..."
    
    # 创建目录
    mkdir -p ~/.gdb/python
    
    # 下载最新版本
    GDB_PY_URL="https://raw.githubusercontent.com/python/cpython/main/Tools/gdb/libpython.py"
    
    if command -v wget >/dev/null 2>&1; then
        wget -O ~/.gdb/python/libpython.py "$GDB_PY_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -o ~/.gdb/python/libpython.py "$GDB_PY_URL"
    else
        echo "需要 wget 或 curl"
        exit 1
    fi
    
    # 创建 gdbinit 配置
    cat > ~/.gdbinit << 'EOF'
# Python gdb 扩展
add-auto-load-safe-path /usr/lib/debug
add-auto-load-safe-path /usr/share/gdb/auto-load

python
import sys
import os

# 添加自定义路径
home = os.path.expanduser("~")
sys.path.insert(0, os.path.join(home, ".gdb/python"))

try:
    import libpython
except ImportError:
    print("无法加载 libpython.py，请检查路径")
end
EOF
    
    echo "源码安装完成，文件保存在 ~/.gdb/python/libpython.py"
}

# 根据操作系统选择安装方法
case $OS in
    ubuntu|debian)
        install_ubuntu
        ;;
    centos|rhel|fedora)
        install_centos
        ;;
    *)
        echo "不支持的操作系统: $OS"
        echo "将尝试从源码安装"
        ;;
esac

# 总是尝试从源码获取以确保可用
install_from_source

echo ""
echo "=== 安装完成 ==="
echo ""
echo "使用示例:"
echo "1. 启动 gdb: gdb /usr/bin/python3"
echo "2. 加载 core 文件: core <corefile>"
echo "3. 在 gdb 中使用 Python 命令:"
echo "   (gdb) py-bt          # 查看 Python 堆栈"
echo "   (gdb) py-list        # 查看 Python 代码"
echo "   (gdb) py-print obj   # 打印 Python 对象"
echo "   (gdb) py-locals      # 查看局部变量"
echo ""
echo "如果自动加载失败，手动加载:"
echo "  (gdb) source ~/.gdb/python/libpython.py"
