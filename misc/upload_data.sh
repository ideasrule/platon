#!/bin/bash

rm data.zip
zip -r data.zip data/
md5sum=$(md5sum data.zip | awk '{print $1}')
mv data.zip data_$md5sum.zip
scp data_$md5sum.zip mz@agn:/home/agn/platon/
echo "Put this md5sum into __init__.py: $md5sum"
