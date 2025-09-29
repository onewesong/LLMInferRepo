cat repo.list | while read line; do
    git clone --depth 1 https://github.com/$line
done