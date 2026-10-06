# 独立备份持久回执 v1

此合同用于可选管理生命周期模式，不改变业务检索、模型池配置或旧备份接口。生产平台消费端需先显式支持 `backup-operation-receipts.v1`，不能只根据 `backup.v1` 推断新接口可用。

## 身份、请求与路由

- 新建：`POST /management/v1/backup-operations`。
- 查询：`GET /management/v1/backup-operations/{operation_id}`。
- 独立管理监听器，仅回环访问，使用独立 Bearer 令牌；不得放进前端。
- 仅在生命周期启用且实际存储格式初始化完成后声明能力。

请求有且仅有以下四个字符串字段；禁止多余字段、类型转换、重复 JSON 键和非有限数，正文上限 16 KiB。

```json
{
  "operation_id": "backup-action-0001",
  "instance_id": "sirchmunk-example",
  "service_instance_id": "00000000-0000-4000-8000-000000000001",
  "maintenance_operation_id": "maintenance-owner-0001"
}
```

此处 ID 仅作示例；调用方必须从已核实的当前实例和维护状态取值。动作 ID 和 owner 都匹配 `^[a-zA-Z0-9-]{8,100}$`；instance 长度 1–80，service_instance 长度 1–100。新动作必须精确匹配当前 instance/启动 UUID、原维护 owner、已知零 HTTP/后台活动、ready 存储且无未知锁。请求中不接受路径、备份名、下载 URL 或命令。

动作 ID 独立于维护 owner。同一维护期可用不同动作 ID 创建不同快照。摘要不作为请求字段提交，而由产品用如下 Python 定义计算，并放入回执；与 flush 相同，与配置接口的 Unicode 编码规则不同：

```python
hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()
```

## 严格 12 字段回执

| 字段 | accepted | succeeded |
| --- | --- | --- |
| `schema_id` | `sirchmunk-backup-operation-result.v1` | 相同 |
| `operation_id` | 原动作 ID | 相同 |
| `instance_id` | 原实例 ID | 相同 |
| `service_instance_id` | 原启动 UUID | 相同 |
| `maintenance_operation_id` | 原维护 owner | 相同 |
| `request_sha256` | 四字段规范请求的 64 位小写 SHA256 | 相同 |
| `state` | `accepted` | `succeeded` |
| `backup_id` | `null` | 内部生成的 `backup-` 加 32 位小写 hex |
| `completed` | `null` | `true` |
| `consistent` | `null` | `true` |
| `data_schema` | `null` | `sirchmunk-data-v1` |
| `effect_committed` | `null` | `true` |

五个结果字段必须一起从 null 转为成功值。accepted 不承诺副作用未发生，不能据此重做；不返回物理路径、错误堆栈、凭据或可下载快照。

处理顺序：锁内检查准入 → FULL SQLite 持久 accepted 并冻结写入 → 内部分配 backup_id → 真实 flush/复制/SQLite online backup → 文件 hash/大小、文件和全部目录祖先刷盘、同卷原子发布 → FULL SQLite 持久 succeeded。GET 不执行快照或 flush；物理复制期间可查询 accepted。若在线 SQLite 备份持有账本锁，GET 可能短暂等待，不承诺恒定响应延迟。

## 幂等与未知状态

账本主键是 `(kind, operation_id)`，新 kind 固定为 `backup-operation`。同 kind/ID/摘要只读原回执，即使已重启或原 UUID 过时；摘要不同返回409。同 ID 在旧 `backup`、`flush` 和新 kind 中互不串单。读取旧成功回执仅用于对账，不能授权新代进程执行后续步骤。

accepted 落盘后才开始快照。accepted 插入失败不执行快照；复制、flush、fsync、发布、成功回执持久化等任一步失败均保留保护状态，不推断成败、不自动重放。目录可能尚无文件、存在 `.partial`，也可能已完整发布；不能用目录存在替代成功回执。未决生命周期回执锁跨 kind 生效；配置 unknown 同样阻止新备份。

客户端中断/取消等待不取消真实 worker，也不提前释放写入锁或 worker 计数。请求超时、断线或 ACK 丢失后，消费者仅 GET 原 ID：

- succeeded：复核 schema、四身份、摘要和五结果，再记录该次完成；下一步重新预检当前进程。
- accepted：保持维护及平台 needs_attention，不发新备份、flush、配置 apply、离开维护或 stop。
- 404/鉴权或连接错误：不是“肯定未执行”，保留原请求身份并人工只读对账，不自动重发。

快照保留备份动作当时的 accepted 账本、原 owner 和进程标记。复制恢复只允许演练到全新隔离 root，恢复后应是 unknown；本接口不实现在线恢复或人工解除 unknown。不删 SQLite/WAL/marker、不改 owner 来绕过保护。

## HTTP 状态与错误码

| 情形 | 状态 / 码 |
| --- | --- |
| 首次成功、重复成功 POST | 200 / succeeded 回执 |
| 重复 accepted POST（仅查询语义） | 202 / accepted 回执 |
| GET 找到任何状态回执 | 200 / 原回执 |
| GET 未知 ID | 404 / `lifecycle_operation_not_found` |
| 无效 GET ID | 422 / `invalid_operation_id` |
| 缺失、额外、类型或长度错误的请求字段 | 422 / `invalid_backup_operation_request` |
| 非对象 JSON 正文 | 422 / `invalid_request` |
| 非法 JSON、重复键或非有限数 | 400 / `invalid_json` |
| 请求大于 16 KiB | 413 / `request_too_large` |
| 无有效独立管理令牌 | 401 / `unauthorized` |
| 生命周期关闭 | 501 / `business_flush_not_proven` |
| 同 ID 摘要不同 | 409 / `lifecycle_operation_reused` |
| 新动作实例或 UUID 不符 | 409 / `lifecycle_identity_mismatch` |
| 维护 owner 不符 | 409 / `maintenance_owner_required` |
| 存在真实在途工作 | 409 / `business_work_not_drained` |
| 生命周期未决/冻结 | 409 / `lifecycle_operation_locked` |
| 计数未知、未就绪或正在停止 | 409 / `lifecycle_not_ready` |
| 配置结果未知 | 409 / `configuration_unknown_locked` |
| 物理 IO / 回执写入异常 | 503 / `lifecycle_effect_unknown` |

错误正文统一 `{"error":{"code":"..."}}`。多个前置条件同时失败时，以实际检查顺序为准，不保证某个较晚条件的错误优先。

## 旧接口与权限

旧 POST `/management/v1/backup` 仍只接受 `operation_id`，要求等于维护 owner；旧 GET `/management/v1/backup/operations/{operation_id}` 仍只查旧 kind，保持原四字段 `backup_id/completed/consistent/data_schema`。不按某种回执是否存在来猜新旧协议。两个协议使用同一真实快照实现。

快照包含可能带密钥的 `.env` 和模型池配置，只留在受限工作根 `.control/backups`。文件及根权限须由部署账户保护：Windows DACL，Ubuntu uid/mode（私有父目录700）；不将私有根挂成公开静态目录。排除可重建模型、静态资源、工具、日志及历史备份。管理 API 不提供下载。业务路径检查禁止 `.control` 及其祖先递归检索。只支持可靠的本地文件系统；本机测试不能证明物理断电或网络文件系统语义。

## 隔离最小复现

以下测试不要求现用服务，不访问知识库/真实令牌，不调用模型供应商。需要安装项目的测试及存储依赖（包括 pytest、httpx、DuckDB、Parquet 引擎、psutil；Windows 权限检查还需 pywin32）。从源码根执行，先设环境再导入任何产品模块。

Windows PowerShell（`<venv-python>` 换成隔离虚拟环境的解释器）：

```powershell
$backupTestRoot = Join-Path ([IO.Path]::GetTempPath()) ('sirchmunk-backup-test-' + [guid]::NewGuid().ToString('N'))
$env:SIRCHMUNK_WORK_PATH = $backupTestRoot
$env:EMBEDDING_CACHE_DIR = Join-Path $backupTestRoot '.cache/models'
$env:HF_HOME = $env:EMBEDDING_CACHE_DIR
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& '<venv-python>' -m pytest tests/test_backup_receipts.py tests/test_backup_receipts_process.py -q --tb=short
```

Ubuntu shell（`<venv-python>` 通常为隔离 venv 的 `bin/python`）：

```sh
backup_test_root=$(mktemp -d)
export SIRCHMUNK_WORK_PATH="$backup_test_root/work"
export EMBEDDING_CACHE_DIR="$backup_test_root/models" HF_HOME="$backup_test_root/models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src"
<venv-python> -m pytest tests/test_backup_receipts.py tests/test_backup_receipts_process.py -q --tb=short
```

测试自身另建私有临时 root、使用合成令牌及随机回环端口；真实子进程测试以当前解释器执行。测试结束不删除现用数据。Windows 使用无窗口子进程；Ubuntu 正常前台子进程，不依赖 PowerShell、SCM 或盘符。真实 HTTP 测试经一次丢 ACK 代理后查询原 ID，并验证 helper/业务 exit0、clean restart 新 UUID 与旧回执只读。

进程故障测试在 accepted 后或目录发布后直接退出，重启后验证 unknown/owner/账本保留且没有重做快照；恢复演练仅复制到新的临时 root。真实监督停机测试需要稳定墙钟；若 WSL/宿主时间跳变触发 `stop_clock_discontinuity`，记录失败并保留保护，不放宽 TTL 或改系统时间以过测。平台联调、SCM/systemd、签名升级和实负载验收独立于本组测试。隔离验证范围见 [Backend validation](backend-validation.md)。
