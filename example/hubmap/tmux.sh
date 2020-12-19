#!/bin/sh
tmux new-session -d -x "$(tput cols)" -y "$(tput lines)" 'export TERM=screen-256color && vim .'
tmux split-window -h -p 35 'watch -n0.1 nvidia-smi'
tmux split-window -v -p 50 'top'
tmux split-window -v 
tmux -2 attach-session -d
