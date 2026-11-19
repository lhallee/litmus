#!/usr/bin/env bash
# Record the desktop footage for the ad on a machine with Docker, from the Linux build of the app, measure the app's memory
# on a long document, and copy the results back.
#
#   capture/run_linux.sh <ssh key> <user@host> <litmus-app (the Linux build with the release built in)> <long text file> <out folder here>
#
# On the machine this uses ~/litmus_run/demo/ only, one container named litmus-demo-capture-* at a time, 6 CPU cores,
# 12 GB, no GPU. The container runs as an ordinary user on its own virtual display; nothing touches a real desktop.
# The texts it pastes and selects are data/ad/texts/ in this folder's parent; the long text is the document the memory
# scene checks. Writes app.mp4, app.json, quick.mp4, quick.json and memory.json in the out folder.
set -euo pipefail
key="$1"
host="$2"
app="$3"
long="$4"
out="$5"
here="$(cd "$(dirname "$0")" && pwd)"
data="$here/../data/ad/texts"
mkdir -p "$out"

ssh -i "$key" -o StrictHostKeyChecking=accept-new "$host" 'mkdir -p ~/litmus_run/demo/{docker,bin,scripts,texts,clips,home}'
scp -i "$key" "$here/Dockerfile" "$host:litmus_run/demo/docker/Dockerfile"
scp -i "$key" "$here/linux_capture.py" "$host:litmus_run/demo/scripts/"
scp -i "$key" "$app" "$host:litmus_run/demo/bin/litmus-app"
scp -i "$key" "$data/ai_essay.txt" "$data/notes.txt" "$host:litmus_run/demo/texts/"
scp -i "$key" "$long" "$host:litmus_run/demo/texts/long.txt"
ssh -i "$key" "$host" 'chmod +x ~/litmus_run/demo/bin/litmus-app && cd ~/litmus_run/demo/docker && sudo docker build --cpuset-cpus 0-1 -t litmus-demo-capture:1 . > /dev/null'
for scene in app quick memory; do
  ssh -i "$key" "$host" "sudo docker run --rm --name litmus-demo-capture-$scene --user 1000:1000 -e HOME=/home/writer --cpus 6 --memory 12g -v /home/ubuntu/litmus_run/demo:/demo -v /home/ubuntu/litmus_run/demo/home:/home/writer litmus-demo-capture:1 python3 /demo/scripts/linux_capture.py $scene /demo/bin/litmus-app /demo/texts /demo/clips"
  if [ "$scene" = memory ]; then
    scp -i "$key" "$host:litmus_run/demo/clips/memory.json" "$out/"
  else
    scp -i "$key" "$host:litmus_run/demo/clips/$scene.mp4" "$host:litmus_run/demo/clips/$scene.json" "$out/"
  fi
done
echo "footage and memory in $out"
