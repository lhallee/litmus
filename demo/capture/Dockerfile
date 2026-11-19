FROM litmus-linux-dev:1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg mousepad dbus-x11 fonts-noto-core fonts-noto-mono dmz-cursor-theme x11-xserver-utils openbox && rm -rf /var/lib/apt/lists/*
# A named user, because the editor warns when run as root and the session bus needs a password entry.
RUN useradd -u 1000 -m -d /home/writer writer
