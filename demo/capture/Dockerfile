# The ad's capture container: a virtual X display, the tools linux_capture.py drives it with, an editor for the quick-check
# scene (with `gsettings` and dconf, without which the editor ignores the font and word-wrap settings the scene's clicks
# are laid out for), and the libraries the Linux build of the app loads (X11, keyboard, cursor, and Mesa's software OpenGL; no GPU).
# Ubuntu 24.04 by digest (pulled 2026-11-19), so the footage's fonts and cursor stay the same from release to release.
# The image's own `ubuntu` user is uid 1000, which run_linux.sh runs the scenes as.
FROM ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 xvfb xdotool xclip x11-utils x11-apps x11-xserver-utils imagemagick ffmpeg procps \
        mousepad dbus-x11 dconf-gsettings-backend dconf-cli libglib2.0-bin openbox fonts-noto-core fonts-noto-mono dmz-cursor-theme \
        libgl1 libegl1 libgl1-mesa-dri libxkbcommon-x11-0 libxcursor1 libxrandr2 libxi6 libx11-xcb1 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*
