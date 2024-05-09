
# move capsule-internal files outside to allow chezmoi to manage?
mv ~/capsule/code/.vscode ~/.vscode
ln -s ~/.vscode ~/capsule/code/.vscode
# this may not be needed once all chezmoi managed files out of capsule
chezmoi apply