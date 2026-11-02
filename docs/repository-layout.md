# 仓库目录约定

仓库采用“一项目一目录”。各项目保留独立源码、配置、测试与 README。

- `extensions/<project-name>/`：浏览器扩展。
- `desktop/<project-name>/`：桌面应用。
- `userscripts/<project-name>/`：油猴脚本；在加入首个脚本时创建。
- `docs/`：仓库级说明。

当多个项目确有公共代码时，可以创建 `shared/`。当前项目各自管理依赖和运行环境。
