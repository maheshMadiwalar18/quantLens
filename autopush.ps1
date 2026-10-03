param (
    [string]$Message = "Auto-update: minor changes"
)

git add .
git commit -m $Message
# The push will happen automatically because of the post-commit hook, 
# but if the hook is bypassed or fails, we can ensure it pushes here too:
git push origin main
