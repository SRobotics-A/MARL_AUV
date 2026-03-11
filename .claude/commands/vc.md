请按照以下步骤执行版本管理流程：

## 第一步：收集变更信息

并行执行以下命令：
- `git status` —— 查看当前工作区状态
- `git diff HEAD` —— 查看所有未暂存和已暂存的变更内容
- `git log --oneline -5` —— 查看最近5条提交，了解提交风格

## 第二步：向用户确认信息

使用 AskUserQuestion 工具，一次性提问以下内容：

1. **暂存哪些文件？**（多选）
   - 选项根据 `git status` 中实际的 modified/untracked 文件动态生成
   - 始终包含"全部文件 (git add -A)"作为一个选项

2. **本次修改的主要原因是什么？**（中文描述，用于 CHANGELOG.md）
   - 选项示例（根据 diff 内容推断）：
     - 修复 bug
     - 新增功能
     - 调整超参数或配置
     - 重构代码结构
     - 其他（用户自定义）

## 第三步：更新 CHANGELOG.md

在 `CHANGELOG.md` 文件的 **最顶部**（`# Reward 设计变更记录` 标题之前，若不存在该标题则在文件开头），插入以下格式的条目：

```
## [日期] 版本更新

**修改文件：**
- 列出本次暂存的文件

**修改原因：**
用户提供的中文描述，结合 git diff 中的实际变更内容，补充技术细节。

**主要变更：**
- 根据 diff 内容，用中文列出 3~5 条具体的技术改动要点

---
```

日期格式为 `YYYY-MM-DD`，内容必须用中文。

## 第四步：执行 git add

根据用户选择：
- 若选择"全部文件"：`git add -A`
- 若选择特定文件：`git add <file1> <file2> ...`
- 务必同时包含 `CHANGELOG.md`：`git add CHANGELOG.md`

## 第五步：创建 commit

用以下格式创建提交（使用 HEREDOC 避免特殊字符问题）：

```bash
git commit -m "$(cat <<'EOF'
<简短英文标题（命令式，50字以内）>

<用中文补充说明修改原因，1~2句话>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

提交标题参考 `git log` 的风格，保持一致。

## 注意事项

- **不要**在未经用户确认的情况下自动选择文件或编写提交信息
- 若 `git status` 显示没有任何变更，告知用户并停止
- 若 commit 因 pre-commit hook 失败，修复问题后重新创建新 commit，不要使用 `--amend`
- 不要执行 `git push`
