# Sirchmunk Web API 接入 Platform Control

本页说明可选管理面，默认关闭。部署的 `PLATFORM_RELEASE_VERSION` 必须匹配实际代码版本，不能用声明覆盖运行版本。原观察/配置模式保留；另需显式启用生命周期模式，才能在单写工作根内提供真实维护、排空、持久刷盘与一致性备份。外部部署平台负责生产发布、签名、SCM/systemd启停和更新切换；产品接口通过不等于生产全链已验收。模型供应商不在管理范围内。英文概览见 [Backend improvements](backend-improvements.md)。

## 启用边界

受管启动必须由进程监督器在导入 `sirchmunk.api.main` **之前**注入以下环境变量。不要把管理凭据写进工作目录的业务 `.env`、源码、镜像层、前端资源或日志。

| 变量 | 用途 |
| --- | --- |
| `SIRCHMUNK_WORK_PATH` | 必填，显式指向本实例工作目录；管理状态保存在其 `.control/` 下。 |
| `SIRCHMUNK_MANAGEMENT_ENABLED=true` | 启用独立管理监听；缺省关闭。 |
| `SIRCHMUNK_MANAGEMENT_INSTANCE_ID` | 平台分配的实例 ID，长度 1–80，字母/数字开头，其余可为字母、数字、`_`、`.`、`-`。 |
| `SIRCHMUNK_DEPLOYMENT_ID` | 本次部署身份，非空且不超过 100 字符。 |
| `SIRCHMUNK_MANAGEMENT_TOKEN` | 独立 Bearer 凭据，至少 32 字符、无空白，不能与业务 `SIRCHMUNK_API_TOKEN` 相同。 |
| `SIRCHMUNK_MANAGEMENT_PORT` | 独立端口，1–65535；只监听 `127.0.0.1`。 |
| `SIRCHMUNK_MANAGEMENT_CONFIG_ENABLED=true` | 可选，启用 `configuration.v1`；缺省关闭。 |
| `SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED=true` | 可选，启用独占根/共享持久存储/真实刷盘与备份；必须同时开启管理，缺省关闭。 |
| `SIRCHMUNK_RELEASE_ROOT` | 不可变发布槽位绝对路径。源码布局默认识别整个仓库根；安装包必须显式设置，并覆盖实际加载包及解释器环境 `sys.prefix`。不能只指向 `src` 或 `site-packages`。 |
| `SIRCHMUNK_MANAGEMENT_SHUTDOWN_TIMEOUT_SECONDS` | 真实活动排空超时，1–600 秒，缺省 60。超时保留 unclean，不强杀工作线程。固定入口由受保护配置设置此值。 |
| `SIRCHMUNK_MAX_WS_CONNECTIONS` | 部署启动时的 WebSocket 上限；配置存储值为 `null` 时恢复此启动值，缺省 100。受管配置要求 1–1000。 |

必须使用单 Web worker、`reload=false`，给 Web API 与管理面各自分配不冲突的端口，并让同机 Platform Agent 访问管理回环端口。第二个进程不能同时打开同一工作目录的受管状态锁。正式部署优先 Ubuntu；在 systemd 或容器编排中从仓库外的受限凭据源注入管理环境，工作目录作为持久卷挂载。不要把 Windows 盘符、PowerShell 或 Windows 进程管理方式当成 Ubuntu 前提。业务 API 仍需单独配置 `SIRCHMUNK_API_TOKEN`、HTTPS 和来源白名单；管理 Bearer 不代替业务鉴权。

原观察/配置模式仍可使用 uvicorn 命令。要接入平台的受监督停止合同，应使用下节固定入口；直接启动 uvicorn 不提供本地停止意图 helper。先在目标机安装依赖、构建静态前端并部署至 `<work>/.cache/web_static`，如需单端口 UI 再注入 `SIRCHMUNK_SERVE_UI=true`：

```sh
<venv>/bin/python -m uvicorn sirchmunk.api.main:app --host 127.0.0.1 --port <web-port> --workers 1
```

Windows 验证环境将 `<venv>/bin/python` 换为虚拟环境的 `Scripts/python.exe`。必须在导入应用前设置`SIRCHMUNK_WORK_PATH`；源码部署还要指向源码`src`的PYTHONPATH，正式制品可安装包。禁止reload或多worker。未经 Ubuntu/Docker 实机验证，不应把 Windows 测试结果当成生产安装验收。

## 固定监督入口与本地停止协议

固定源码入口为 `scripts/supervised_serve.py`，不依赖 cwd/PYTHONPATH，兼容 Python `-I`，显式禁止写入代码槽位的 `.pyc`。它在前台常驻，不 daemonize、不脱离父进程树。相同入口的 helper 模式只读原进程身份/SQLite 回执及写固定 mailbox，不导入业务路由、不 claim 工作根、不启动第二业务实例。

```sh
<venv>/bin/python -I <release>/scripts/supervised_serve.py --config <protected-config>
<venv>/bin/python -I <release>/scripts/supervised_serve.py --config <protected-config> --check-only
<venv>/bin/python -I <release>/scripts/supervised_serve.py --config <protected-config> --request-stop --stop-timeout-seconds 60 --intent-file <fixed-intent>
```

`--check-only` 校验配置、代码/数据路径和启动身份，不打开存储；它不是存储/模型可用性证明。`PLATFORM_DEPLOYMENT_ID` 可映射为真实部署身份；与显式 `SIRCHMUNK_DEPLOYMENT_ID` 冲突时拒绝。`PLATFORM_RELEASE_VERSION` 若存在必须等于代码中的真实 `__version__`，不能覆盖运行版本。Control 只展开 argv 中的 `{release}`，不要在 environment 中放未展开的占位符。

仓库中的 `config/supervisor.example.json` 是 Ubuntu 路径示例，**必须复制到仓库和工作根之外**、修改为本机受保护的绝对路径后使用。Windows 使用相应绝对路径和 JSON 转义；不把示例路径当成生产默认值。配置严格九键，不接受命令、下载 URL、任意 argv 或额外字段。mailbox 固定为 `<work>/.control/stop-ipc`；intent 必须在工作根及发布槽位外，与配置文件不同，且与平台 stop_command 中唯一 `--intent-file` 参数一致。

配置、intent 和 mailbox/账本必须由服务账户、SYSTEM/root 或 Administrators 控制；目标及直接父目录不允许其他主体写入，全路径拒绝 symlink/reparse，控制文件拒绝硬链接。Windows 通过 `pywin32` 检查 DACL/所有者；仅已验证的可信所有者可使用 OWNER RIGHTS ACE。Ubuntu 使用 uid/mode，不安装 pywin32。更高祖先目录的防替换和专用账户隔离还须由 Control 部署校验负责；本产品不防御同账户恶意程序/管理员。

平台新鲜预检原实例、维护 owner、ready、零 active/queued 并确认持久 flush 后，才生成六字段 `platform-stop-intent.v1`：

```json
{
  "schema": "platform-stop-intent.v1",
  "instance_id": "sirchmunk-example",
  "service_instance_id": "<original verified UUID>",
  "maintenance_owner": "<original verified operation ID>",
  "nonce": "<64 lowercase hex characters>",
  "expires_at": 0
}
```

以上仅为字段示意，`expires_at` 必须是平台生成时 Unix 秒数加 120，不得使用示例 0。helper/业务端均严格校验字段、类型、有效期；业务线程在接受请求的锁内再校验一次，绝不以当前 UUID 替换旧意图。接受前复核原 owner、实际零工作、无 unknown；先 FULL SQLite accepted，再真实 flush，才开始协作关闭。关闭所有受管存储后，同一 FULL transaction 提交 `sirchmunk-stop-result.v1` 成功回执和 unclean=0。回执九键为 schema_id、nonce、instance_id、service_instance_id、maintenance_owner、request_sha256、state、stopped、effect_committed；摘要规则与 flush 相同。accepted 对应 null/null，succeeded 对应 true/true。

helper 只有观察到原 nonce/UUID 成功回执、clean 标记，且原 PID+create_time 已退出才 exit0。超时/失败 exit1，只输出固定错误码，不撤回、不改 nonce、不重发、不强杀。对已提交的相同意图再次调用只查询，即使 TTL 已过也不能重新投递；新意图必须在有效期内。拒绝旧 UUID、错误 owner、损坏回执或未知活动时不向业务进程发停止信号。

同一服务进程持续比较系统 wall clock 与 monotonic 启动基线，偏差超过2秒后锁存 `stop_clock_discontinuity` 并拒绝新停止副作用，不自动重置基线，防止时钟倒退使过期意图复活。最终TTL使用已校验的单调基线时间与墙钟采样的最大值，并保存跨调用不倒退的时间上界；2秒容忍不是过期宽限。已有回执仍可查询。时钟跳变时保留维护/平台错误状态，由管理员先处理时间与进程状态；不能删除账本或重新发nonce绕过。此检查不修改系统时间服务。

Windows helper 可用 `CREATE_NO_WINDOW`，不依赖 CTRL_BREAK/控制台。SCM host 仍须独立确认业务 Job 和 helper Job 全部退出、两个 exit0，才能清 host marker。Ubuntu systemd 的 ExecStop 使用同一 helper；平台合同使用 `KillMode=control-group`、`SendSIGKILL=no`、`KillSignal=SIGCONT`，防止 helper 拒绝后 systemd 默认 SIGTERM 停错实例。仅产品 helper 通过不代表上述系统服务合同验收。本批不安装系统服务、不自动提权。

## 固定源码构建元数据

`scripts/build_release_metadata.py` 使用 Python 3.11+ 和本机 Git，只从完整固定 commit SHA 读取 Git 原始 blob，忽略工作树/CRLF/replace refs，不导入产品、不打包依赖、不下载模型、不签名。默认输出 JSON，`--output` 仅新建文件，不覆盖已有交付物：

```sh
python scripts/build_release_metadata.py --repo <source-repository> --commit <full-commit-sha> --output <new-metadata-file>
```

`sirchmunk-release-build.v1` 包含真实版本、产品 sirchmunk、source_commit、逐文件 mode/size/SHA256、清单摘要、data_schema、migrations=none，以及实际构建机的 OS/arch/Python ABI 和声明的 Python 范围。它拒绝私有运行目录/数据库/凭据路径、符号链接和子模块；不读取未跟踪文件。源码内仍不能提交密钥，路径拒绝不代替秘密扫描。构建机 ABI 是事实，不是跨平台已验收声明；Ubuntu/Windows 运行制品分别构建并验证依赖。源码包不含 venv、模型、业务配置、知识库、`.control` 或 wheel。

此元数据不是 Control 签名发布 envelope。正式发布必须由平台明确支持 `platform_type=sirchmunk`、schema 和迁移策略并完成签名/审批/槽位及回退校验；不得伪装其他产品类型绕过白名单。

## 管理合同

所有 `/management/v1` 请求使用独立的 `Authorization: Bearer <management-token>`；未授权返回 401。管理端与业务端是不同监听器。

| 路由 | 行为 |
| --- | --- |
| `GET /management/v1/capabilities` | 返回 `health.v1`、`identity.v1`、`maintenance.v1`、`persistent-maintenance.v1`、`drain.v1`；仅在配置开关打开时包含 `configuration.v1`。 |
| `GET /management/v1/status` | 返回真实版本、部署/启动身份、维护门、Web 进程内 active/queued 与是否接单。异常退出遗留未完成预约时计数为 `null`、readiness 为 `unknown`，并拒绝新业务。 |
| `POST /management/v1/maintenance` | 以 `enabled` 布尔值及 `operation_id` 设置持久 owner。相同 owner 重试幂等；其他 owner 409。它只排空本 Web 进程，不意味着可以安全自动重启。 |
| `GET /management/v1/config` | 返回 `sirchmunk.runtime.v1` 的 revision、stored、真实 consumer 的 effective 和 apply state。 |
| `POST /management/v1/config/validate` | 只校验严格字段、值、摘要和 revision，不写状态。 |
| `POST /management/v1/config/operations` | CAS 应用一次并持久记录回执；同一操作 ID 不会重复执行。 |
| `GET /management/v1/config/operations/{operation_id}` | 只读取得原回执；请求 ACK 丢失时应先查此接口。 |
| `GET /management/v1/readiness` | 只读存储验证、后台活动与不可就绪原因，不增加原status的14字段。 |
| `POST /management/v1/flush/operations` | 生命周期模式的持久四字段请求；真实持久屏障后返回九字段成功回执。 |
| `GET /management/v1/flush/operations/{operation_id}` | 只读原回执，支持跨进程重启查询，不重跑。 |
| `POST /management/v1/flush {}` | 兼容路由，要求当前维护owner、零活动、无未知锁；内部生成持久操作。超时不能重发。 |
| `POST /management/v1/backup` | 旧接口不变：请求仅operation_id（必须等于维护owner），返回旧四字段结果。 |
| `GET /management/v1/backup/operations/{operation_id}` | 只读旧四字段备份结果；未完成回执不表示副作用未发生。 |
| `POST /management/v1/backup-operations` | 独立动作ID、四字段请求、12字段持久回执；同一维护owner可创建多份不同快照。 |
| `GET /management/v1/backup-operations/{operation_id}` | 只读新备份回执；包括accepted在内，找到统一200，未知ID404。 |

关闭生命周期开关时flush/backup（包括新backup-operations）仍为501，且不声明对应能力。实际存储格式验证后才报告data_schema=`sirchmunk-data-v1`和`flush.v1`、`flush-operation-receipts.v1`、`backup.v1`、`backup-operation-receipts.v1`。健康但维护中的readiness仍为`ok`、accepting_jobs=false。

## 生命周期请求、持久性与未知态

工作根持有OS独占锁；升级后的CLI/MCP/SDK不能同时写，也不能在Web停机期间独立打开已受管根。旧二进制或同用户任意程序不遵守本锁，首次接管须停旧程序并配置专用OS账户/工作根权限。本模式仅支持本地可靠文件系统，不宣称网络文件系统或硬件断电实测。数据根不得处于代码release槽位中。

维护关闭新HTTP/SSE/WS准入；既有WS当前消息可以结束，空闲连接与下一轮消息关闭1013。状态包含请求与真实后台lease（知识演化、cleanup、模型线程、子进程）。客户端取消不意味着这些工作已经退出，flush只在实际计数归零后执行。

刷盘请求有且仅有`operation_id`、`instance_id`、`service_instance_id`、`maintenance_operation_id`。摘要为UTF-8的`json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)`的SHA256。回执有且仅有`schema_id`（`sirchmunk-flush-operation-result.v1`）、前述四字段、`request_sha256`、`state`、`flushed`、`effect_committed`。先FULL同步SQLite记录accepted（null/null），再对共享DuckDB/Parquet及权威文件与全部目录祖先刷盘；最后持久succeeded（true/true）。同ID同摘要只读原回执，摘要不同409。旧UUID的成功回执可查询，但不能授权新进程继续旧切换。

accepted崩溃、刷盘错误或成功ACK未知时，不能假定未生效；只查询原ID，不重发POST。未知回执保留维护与写门，不能通过新ID、配置apply、离开维护、删SQLite/WAL/marker来解锁。管理GET在物理刷盘期间仍可响应。

备份同样先持久accepted，冻结业务及管理mutator后真实flush。快照包含权威业务文件（包括可能含密钥的.env/模型池配置）、schema marker、SQLite online backup、文件SHA256/大小清单；排除模型/静态资源/tools/logs和历史/临时备份。旧接口成功结果仍为`backup_id`、`completed=true`、`consistent=true`、`data_schema`四字段。快照发布后才写成功回执，所以快照中的维护owner与accepted账本保留；还原到新root后应保持unknown，必须人工只读对账，禁止删账本自动开门。快照继承私有root权限，不可加入源码包/镜像或公开日志。受管业务文件路径检查拒绝直接访问`.control`或递归扫描其祖先目录；请指定data/uploads或独立语料子树，不把整个工作根作为检索路径。

新`backup-operations`严格接受与flush相同的四个身份字段，动作ID独立于维护owner，摘要规则同flush。12字段回执包含四身份字段、`schema_id=sirchmunk-backup-operation-result.v1`、`request_sha256`、`state`以及`backup_id/completed/consistent/data_schema/effect_committed`。accepted时后五字段全为null；真实快照发布及回执持久化完成后才变为succeeded、新backup_id、三个true和`sirchmunk-data-v1`。ledger kind固定为`backup-operation`，与旧`backup`和`flush`相互独立，不按存在性猜版本。任何kind的未决回执或配置unknown都阻止新备份。

新POST成功200，重复accepted请求只读返回202；GET找到回执统一200，包括accepted。坏JSON/重复键400，非法字段/值422，超16KiB为413，鉴权失败401，冲突/未排空/unknown锁409；物理IO或回执失败503，仅返回固定码`lifecycle_effect_unknown`。客户端取消等待不释放仍在执行的worker及写入锁；ACK丢失后仅GET原ID，不重发或换ID绕过。旧UUID原回执可查询，但新动作必须绑定当前进程且不能凭旧成功回执授权。完整字段、错误码、跨平台最小复现见[独立备份回执协议](backup-operation-receipts.md)。

受管Embedding只读预置本地模型，不在线下载；缺失模型导致对应操作明确失败。存储ready不证明远程LLM可用，不以测试请求真实供应商来验证ready。

配置请求必须有且仅有 `protocol_version`、`operation_id`、`instance_id`、`service_instance_id`、`config_schema_id`、`expected_revision`、`changes`、`request_digest`。协议版本为 `1.0`，schema 为 `sirchmunk.runtime.v1`；`changes` 仅支持 `maxWebSocketConnections`，值须是严格整数 1–1000 或 `null`。`null` 表示恢复本次启动时的部署上限。摘要是去掉 `request_digest` 后，对对象按键排序、无空格分隔、UTF-8、保留非 ASCII 字符的规范 JSON 求 SHA-256。请求正文上限 16 KiB，重复 JSON 键和非有限数拒绝。

降低上限不主动断开存量 WebSocket；超限新连接以 1013 关闭。若提交结果未知，不要重发新操作或清理 `.control/`：先按原 operation ID 查询回执、读取配置和状态，再由管理员只读对账。持久 `unknown` 会锁住后续配置；不能凭一次超时推断生效或失败。

## 回退与验收

生命周期root仅允许回退到理解同一data_schema、独占锁和持久回执的代码，不能关闭开关后用老版程序强开受管root；只回退代码，不自动回滚业务数据。保留业务`.env`、知识库、模型池配置和`.control/`供审计；不要删除状态文件解除unknown或维护owner。观察/配置旧模式的回退规则不代表生命周期root兼容旧二进制。源码交付包只含源码和补丁，不含凭据、运行数据、wheel或已构建静态资源；目标机必须构建并在独立目录校验、演练回退。

隔离验证范围与可复现命令见 [Backend validation](backend-validation.md)。SCM/systemd部署、发布签名、跨主机平台联调、真实负载与升级/回退演练需在目标环境另行验收；本接口测试不替代这些部署门槛。
