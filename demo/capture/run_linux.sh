#!/usr/bin/env bash
# Record the desktop footage for the ad with Docker, from the Linux build of the app, measure the app's memory on a long
# document, and copy the results back.
#
#   capture/run_linux.sh <ssh key> <user@host> <litmus-app (the Linux build with the release built in)> <long text file> <out folder here>
#   capture/run_linux.sh local <stage folder> <litmus-app> <long text file> <out folder here>
#
# Over SSH it uses the machine's ~/litmus_run/demo/ only and `sudo docker`. With `local` it stages in <stage folder> and
# uses this machine's `docker` (Docker Desktop on Windows runs the x86_64 build natively; Logan allowed this desktop for
# short Docker jobs on 2026-11-19). Either way: one container named litmus-demo-capture-* at a time, 6 CPU cores, 12 GB,
# no GPU, run as an ordinary user on its own virtual display; nothing touches a real desktop. The texts it pastes and
# selects are data/ad/texts/ in this folder's parent; the long text is the document the memory scene checks. Writes
# app.mp4, app.json, quick.mp4, quick.json and memory.json in the out folder.
set -euo pipefail
app="$3"
long="$4"
out="$5"
here="$(cd "$(dirname "$0")" && pwd)"
data="$here/../data/ad/texts"
image="litmus-demo-capture:2"
mkdir -p "$out"

if [ "$1" = local ]; then
  mkdir -p "$2"
  root="$(cd "$2" && pwd)"
  mount="$root"
  if command -v cygpath > /dev/null; then
    mount="$(cygpath -m "$root")"  # Docker Desktop wants C:/..., and Git Bash must not rewrite /demo paths
    export MSYS_NO_PATHCONV=1
  fi
  docker=docker
  on_host() { (cd "$root" && bash -c "$1"); }
  put() { cp "$1" "$root/$2"; }
  get() { cp "$root/$1" "$2"; }
else
  key="$1"
  host="$2"
  ssh -i "$key" -o StrictHostKeyChecking=accept-new "$host" 'mkdir -p ~/litmus_run/demo'
  mount="$(ssh -i "$key" "$host" 'cd ~/litmus_run/demo && pwd')"
  docker="sudo docker"
  on_host() { ssh -i "$key" "$host" "cd ~/litmus_run/demo && $1"; }
  put() { scp -i "$key" "$1" "$host:litmus_run/demo/$2"; }
  get() { scp -i "$key" "$host:litmus_run/demo/$1" "$2"; }
fi

on_host 'mkdir -p docker bin scripts texts clips home'
put "$here/Dockerfile" docker/Dockerfile
put "$here/linux_capture.py" scripts/linux_capture.py
put "$app" bin/litmus-app
put "$data/ai_essay.txt" texts/ai_essay.txt
put "$data/notes.txt" texts/notes.txt
put "$long" texts/long.txt
on_host "chmod +x bin/litmus-app && $docker build --cpuset-cpus 0-1 -t $image docker > /dev/null"
for scene in app quick memory; do
  on_host "$docker run --rm --name litmus-demo-capture-$scene --user 1000:1000 -e HOME=/home/writer --cpus 6 --memory 12g -v $mount:/demo -v $mount/home:/home/writer $image python3 /demo/scripts/linux_capture.py $scene /demo/bin/litmus-app /demo/texts /demo/clips"
  if [ "$scene" = memory ]; then
    get clips/memory.json "$out/"
  else
    get "clips/$scene.mp4" "$out/"
    get "clips/$scene.json" "$out/"
  fi
done
echo "footage and memory in $out"
