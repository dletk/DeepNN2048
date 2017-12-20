#!/usr/bin/env bash
for (( i = 0; i < 100; i++ )); do
  echo "======> Running time #${i}"
  echo `python3 GameManager_3.py`
  echo "==> Finished time #${i}"
done
