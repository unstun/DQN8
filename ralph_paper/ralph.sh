#!/bin/bash
# Ralph Paper Rewriter — 自主 SCI 论文改进循环
# 用法: ./ralph_paper/ralph.sh [max_iterations]
# 从项目根目录运行: /home/sun/phdproject/dqn/DQN8/

set -e

MAX_ITERATIONS=${1:-15}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$SCRIPT_DIR/CLAUDE.md"

# 8小时时间限制
START_TIME=$(date +%s)
MAX_SECONDS=$((8 * 3600))

echo "============================================="
echo "  Ralph Paper Rewriter"
echo "  项目: $PROJECT_DIR"
echo "  PRD: $SCRIPT_DIR/prd.json"
echo "  最大迭代: $MAX_ITERATIONS"
echo "  时间限制: 8 小时"
echo "  开始: $(date)"
echo "============================================="

for i in $(seq 1 $MAX_ITERATIONS); do
  # 检查 8 小时限制
  ELAPSED=$(( $(date +%s) - START_TIME ))
  if [ $ELAPSED -ge $MAX_SECONDS ]; then
    echo ""
    echo "已达 8 小时时间限制，停止。"
    echo "已运行: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
    exit 0
  fi

  REMAINING=$(( MAX_SECONDS - ELAPSED ))
  echo ""
  echo "==============================================================="
  echo "  Ralph 迭代 $i / $MAX_ITERATIONS"
  echo "  剩余时间: $(( REMAINING / 3600 ))h $(( (REMAINING % 3600) / 60 ))m"
  echo "==============================================================="

  cd "$PROJECT_DIR"
  OUTPUT=$(claude --dangerously-skip-permissions -p "$(cat "$PROMPT_FILE")" 2>&1 | tee /dev/stderr) || true

  # 检查完成信号
  if echo "$OUTPUT" | grep -q "COMPLETE"; then
    echo ""
    echo "所有论文任务完成！"
    echo "在第 $i 轮完成，共 $MAX_ITERATIONS 轮"
    echo "总耗时: $(( ($(date +%s) - START_TIME) / 60 )) 分钟"
    exit 0
  fi

  echo "第 $i 轮完成。5 秒后继续..."
  sleep 5
done

echo ""
echo "已达最大迭代次数 ($MAX_ITERATIONS)。"
echo "查看 $SCRIPT_DIR/progress.txt 了解进度。"
exit 1
