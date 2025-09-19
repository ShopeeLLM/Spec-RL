#!/bin/bash
# fix script for vllm 0.9.1 oov issue
# https://github.com/volcengine/verl/issues/1398
# https://github.com/vllm-project/vllm/issues/13175
# replace vllm processor.py to current processor.py
PROJECT_DIR=$1
SRC="${PROJECT_DIR}/fix_oov/processor.py"
# checkout your own vllm package place
# for docker image provided by: verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2
# using this directly
DST="/usr/local/lib/python3.10/dist-packages/vllm/v1/engine/processor.py"

# check if source file exists
if [ ! -f "$SRC" ]; then
    echo "❌ source $SRC not exist"
    exit 1
fi

# check if target file exists
if [ ! -f "$DST" ]; then
    echo "❌ target $DST not exist"
    exit 1
fi

# backup target file first 
cp "$DST" "${DST}.bak"
echo "✅ backup target file to ${DST}.bak"

cp "$SRC" "$DST"
echo "✅ replaced $DST"

# replace vllm llm_engine.py to current llm_engine.py

SRC="${PROJECT_DIR}/fix_oov/llm_engine.py"
DST="/usr/local/lib/python3.10/dist-packages/vllm/engine/llm_engine.py"

# check if source file exists
if [ ! -f "$SRC" ]; then
    echo "❌ source $SRC not exist"
    exit 1
fi

# check if target file exists
if [ ! -f "$DST" ]; then
    echo "❌ target $DST not exist"
    exit 1
fi

# backup target file first 
cp "$DST" "${DST}.bak"
echo "✅ backup target file to ${DST}.bak"

cp "$SRC" "$DST"
echo "✅ replaced $DST"
