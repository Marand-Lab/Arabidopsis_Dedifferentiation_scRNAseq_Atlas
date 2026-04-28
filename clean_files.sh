grep -rlE 'alex|amarand' . \
  --exclude-dir=.git \
  --exclude='*.png' --exclude='*.jpg' --exclude='*.pdf' \
| xargs sed -i '' -E 's/alex|amarand/YOURNAME/g'
