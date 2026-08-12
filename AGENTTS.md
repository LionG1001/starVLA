# Codex Kubernetes 远端开发约束

## 工作目录映射

- 本地同步根目录：`C:\works\jd_2026\vla_project\old_starvla\starVLA`。
- 远端同步根目录：`/home/jd/gl_dev/starVLA`。
- 代码基线：GitLab 的 StarVLA MUSA 适配代码。
- 代码修改与测试优先在远端 Kubernetes 工作负载中执行；完成后将代码改动同步回本地，供 Review 和 Git 管理。

## K8s 远端开发环境

- 通过 `bastion-k8s` 访问集群，不直接从本地 SSH 到 Pod 或节点。
- Namespace：`his-test`。
- 工作负载类型：Deployment。
- Deployment：`jd-starvla-ws128-test`。
- Pod 标签选择器：`app=jd-starvla-ws128-test`。
- 容器名：`his-test`。
- Pod 名会随 Deployment 更新而变化，不得把某个具体 Pod 名写死到脚本中；每次从 Running Pod 中动态选择。
- 容器内工作目录：`/home/jd/gl_dev/starVLA`。
- `/home/jd` 为多副本共享存储，代码只需同步到任意一个 Running Pod 的该路径一次。

## 访问与验证流程

1. 首次操作先执行 bastion 连通性检查，再验证 Kubernetes API 可达。
2. 在 `his-test` namespace 中按标签选择一个 Running Pod，并显式指定 `his-test` 容器。
3. 执行任务前运行 `cd /home/jd/gl_dev/starVLA && pwd`，返回值必须与约定路径完全一致。
4. 若没有 Running Pod、容器名变化或共享挂载不可用，停止同步或测试并报告问题，不得改用未确认的 Pod、节点或路径。

参考命令：

```bash
NS=his-test
APP=jd-starvla-ws128-test
CONTAINER=his-test
POD=$(kubectl get pod -n "$NS" -l "app=$APP" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n "$NS" -c "$CONTAINER" "$POD" -- \
  bash -lc 'cd /home/jd/gl_dev/starVLA && pwd'
```

## 同步约束

- 同步前分别检查本地和远端目标范围；远端存在未知文件或双方同一路径均有修改时，不得直接覆盖。
- 默认同步代码、配置、文档和测试文件，并保持相对路径一致。
- 默认排除 `.git`、`.mccl_error*`、模型权重、数据集、检查点、日志、缓存、虚拟环境、构建产物和测试输出。
- 文本文件使用 LF；遵守仓库根目录 `.gitattributes`。
- 同步完成后核对文件数量、总字节数和逐文件内容哈希，并检查本地 `git status --short`。
- 未经用户明确要求，不得提交、推送、修改 Deployment、清理 Pod、终止进程或改动集群资源。

## 结果报告

- 报告实际使用的 namespace、Deployment、Pod、容器和工作目录。
- 报告同步范围、排除项、文件校验结果以及执行过的测试。
- 未完成或未验证的结果不得描述为成功。


# Codex 远端开发约束

## 内网远端容器开发环境

- 本地项目目录：`C:\works\jd_2026\vla_project\old_starvla\starVLA`。
- 远端主机：`10.20.35.29`。
- SSH 用户名：`mccxadmin`。
- 远端容器：`gl-vla`。
- 容器内工作目录： `/data/share/liang.geng/starvla_work/starVLA`。
- SSH 认证优先使用 SSH Key；必须使用密码时，只能通过交互式提示或当前进程的临时环境变量提供。不得将密码写入本文件、脚本、命令参数、日志、Shell 历史或其他持久化配置，也不得在输出中回显密码。

## 远端访问与范围

- 仅在用户要求远端开发、运行或测试时连接上述主机，并始终启用 SSH 主机密钥校验；首次连接可以使用 `StrictHostKeyChecking=accept-new`，不得使用 `StrictHostKeyChecking=no`。
- 执行任务前，先确认 `gl-vla` 容器存在且正在运行，并通过 `docker exec -w /data/share/liang.geng/vla-project gl-vla pwd` 验证容器内工作目录与约定完全一致。
- 所有远端代码修改、构建和测试均应在 `gl-vla` 容器的上述工作目录中执行。不得在宿主机的其他目录、其他容器或项目范围之外进行修改。
- 不得擅自修改远端服务、系统配置或无关项目，也不得执行破坏性操作；确有需要时必须先获得用户明确授权。

## 本地与远端同步

- 默认不自动同步整个仓库。只有当用户明确指定需要同步的目录或文件时，才建立本地与远端的同步范围。
- 同步时保持相对于项目根目录的路径一致：本地根目录对应 `C:\works\jd_2026\vla_project\starVLA`，远端根目录对应容器内 `/data/share/liang.geng/vla-project`。
- 开始修改前，检查本地和远端目标范围的 Git 状态及已有改动。若同一文件两端均存在未同步修改，暂停覆盖并向用户报告冲突。
- 对于用户指定的同步范围，先确保远端具有本地基线，然后在远端容器内完成代码修改和测试。
- 远端修改完成后，必须将变更同步回本地对应路径，便于用户在本地 Review；同步操作不得覆盖同步范围之外的文件。
- 同步时默认排除 `.git`、虚拟环境、缓存、数据集、模型权重、检查点、构建产物和测试输出，除非用户明确要求同步其中某项。
- 同步完成后，使用 `git status --short`、`git diff --check` 以及必要的文件校验确认本地变更完整且无意外文件。本地工作树中的同步结果是交付用户 Review 的版本。
- 未经用户明确要求，不得自动提交、推送或创建 Pull Request。

## 结果报告

- 每次远端任务结束时，报告实际使用的主机、容器、工作目录、修改和同步的文件范围、执行的测试及测试结果。
- 若测试无法运行或同步不完整，明确说明原因和仍待处理的内容，不得将未验证结果描述为已完成。
