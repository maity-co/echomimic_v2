#! /bin/bash
dir_to_sync='echomimic_v2'
usr='web'
kraken='192.168.0.48'
dest_dir='/home/web/dev'

while inotifywait -r --exclude '/\.' ../$dir_to_sync/*; do
  rsync --exclude=".*" --exclude="uploads"  -av ../$dir_to_sync/ $usr@$kraken:$dest_dir/$dir_to_sync
#  sleep 2
#  echo "second sync."
#  rsync --exclude=".*"  --exclude="uploads"  -av ../$dir_to_sync/ $usr@$kraken:$dest_dir/$dir_to_sync
done
