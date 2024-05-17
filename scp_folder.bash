#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: bash scp_folder.bash <folder_name>"
  exit 1
fi

FOLDER_NAME=$1
REMOTE_USER="xix22010"
PROXY_JUMP="xix22010@137.99.0.102"
REMOTE_HOST="192.168.10.16"
REMOTE_BASE_PATH="/home/xix22010/py_projects/from_azure"

ssh -o ProxyJump=$PROXY_JUMP $REMOTE_USER@$REMOTE_HOST "if [ -d $REMOTE_BASE_PATH/$FOLDER_NAME ]; then exit 0; else exit 1; fi"

if [ $? -eq 0 ]; then
  echo "Remote folder exists. Copying files..."
else
  echo "Remote folder does not exist. Do you want to create it? (yes/no) [yes]: "
  read answer
  answer=${answer:-yes}
  if [ "$answer" != "yes" ]; then
    echo "Operation aborted."
    exit 1
  fi
  
  ssh -o ProxyJump=$PROXY_JUMP $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_BASE_PATH/$FOLDER_NAME"
fi

FILES=("acc.txt" "args.txt" "best_model.pth" "cmd.txt")

for FILE in "${FILES[@]}"; do
  LOCAL_FILE="$FOLDER_NAME/$FILE"
  REMOTE_FILE="$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE_PATH/$FOLDER_NAME/$FILE"
  
  if [ -f "$LOCAL_FILE" ]; then
    scp -o ProxyJump=$PROXY_JUMP -o "StrictHostKeyChecking=no" "$LOCAL_FILE" "$REMOTE_FILE"
  else
    echo "Warning: $LOCAL_FILE does not exist locally and will not be copied."
  fi
done

echo "Files copied successfully."
